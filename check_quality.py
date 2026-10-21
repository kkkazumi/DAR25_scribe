#!/usr/bin/env python3
"""
文字起こしの品質チェック表を作る（PyCharm用）。音声は使わず、すでにあるCSVだけで計算します。

前提: align_questions.py まで済んでいる（out_aligned/*_aligned.csv ができている）。
あれば使うもの:
    out_transcript/<mp3名>_transcript.csv  Whisperの警告（無音の疑い・低確信・定型句など）
    out_transcript/<mp3名>_words.csv       単語ごとの確信度
    out_overlap/<mp3名>/utterance_overlap.csv  発話ごとの重なり率（detect_overlap.py の出力）

出力（out_quality/）:
    <mp3名>_quality.csv   発話ごとの品質指標と「要確認」の理由
    quality_summary.csv   ファイルごとの集計

指標:
    食い違い度      WhisperとAzureの文字の差（0=一致〜1=全然違う）。句読点とフィラー、カタカナ/ひらがなの違いは無視。
                    漢字⇔ひらがなの表記ゆれ（「分かる」「わかる」「通る」「とおる」）は、差として数えない（辞書は使わない簡易判定）。
    単語確信度      Whisperが出した単語ごとの確信度の平均と最小
    Whisperの警告   transcribe.py が付けた旗（無音の疑い・低確信・定型句・除去あり など）
    重なり率        その発話のうち、2人が同時に話していた時間の割合
    文の不自然さ    前の発話の末尾が混ざっていそう／途中で切れていそう、などの疑い

使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import difflib
import re
import statistics
import sys
from pathlib import Path

from align_questions import normalize
from transcribe import FILLER_RE

# ============== 【設定】 ==============
TARGETS = []                 # 処理する mp3 名（拡張子なし）。[] なら out_aligned 内のすべて
EXCLUDE = ["DAR25_007_Human", "DAR25_008_Human", "DAR25_031_Human", "DAR25_047_Human"]   # 話者が1人だけで分けられないファイル（対応後に外す）
MISMATCH_FLAG = 0.30         # 食い違い度がこれ以上なら「食い違い大」
CONF_MEAN_FLAG = 0.70        # 単語確信度の平均がこれ未満なら「低確信」
CONF_MIN_FLAG = None         # 単語確信度の最小がこれ未満なら「低確信(単語)」。None なら使わない（長い発話では、必ず1語は低くなるため）
OVERLAP_FLAG = 0.10          # 重なり率がこれ以上なら「重なり」
MIN_CHARS_FOR_COMPARE = 4    # これより短い発話は食い違い度を出さない（短すぎて不安定）
# =====================================

BASE = Path(__file__).resolve().parent
ALIGNED_DIR = BASE / "out_aligned"
TRANS_DIR = BASE / "out_transcript"
OVERLAP_DIR = BASE / "out_overlap"
OUT_DIR = BASE / "out_quality"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def comparable(text: str) -> str:
    """句読点・フィラー・カタカナ/ひらがなの違いを無視した形にする"""
    return normalize(FILLER_RE.sub("", text))


_KANJI = re.compile(r"[一-龥々〆ヵヶ]")
_KANA_ONLY = re.compile(r"^[ぁ-ゖー]+$")


def _is_script_variant(x: str, y: str) -> bool:
    """片方が漢字を含み、もう片方がひらがなだけ（読みの表記ゆれ）か。例: 「分」⇔「わ」「通」⇔「とお」"""
    for kanji_side, kana_side in ((x, y), (y, x)):
        n = len(_KANJI.findall(kanji_side))
        if n and _KANA_ONLY.match(kana_side) and len(kana_side) <= 4 * n and len(kanji_side) <= 6:
            return True
    return False


def mismatch(whisper: str, azure: str) -> float | None:
    """食い違い度。漢字⇔ひらがなの表記ゆれ（読みが違うだけ）は、差として数えない。"""
    a, b = comparable(azure), comparable(whisper)
    n = max(len(a), len(b))
    if n < MIN_CHARS_FOR_COMPARE:
        return None
    cost = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        x, y = a[i1:i2], b[j1:j2]
        if tag == "replace" and _is_script_variant(x, y):
            continue
        cost += edit_distance(x, y)
    return min(1.0, cost / n)


# --- 文の不自然さ ---
LEAD_QUESTION_TAIL = re.compile(r"^.{0,10}?(ですか|ますか|でしょうか|ですかね|ますかね|か)[。？?]")
TRAILING_PARTICLES = ("を", "に", "が", "は", "も", "の", "と", "で", "て", "から", "ので", "けど", "けれど", "し", "や", "へ")
LEAD_PARTICLES = ("を", "が", "の", "へ", "に")  # 文頭に来にくい助詞（前の発話の続きの疑い）


def oddness(text: str, role: str) -> list[str]:
    t = text.strip()
    if not t:
        return ["空"]
    flags = []
    body = re.sub(FILLER_RE.pattern + r"[、。,.]?", "", t)  # 先頭のフィラーは除いて判定
    m = LEAD_QUESTION_TAIL.match(body)
    if m and len(body) > m.end() + 2:
        flags.append("文頭が前の質問の末尾の疑い")
    elif body[:1] in LEAD_PARTICLES and len(body) > 4:
        flags.append("文頭が助詞（前の発話の続きの疑い）")
    tail = re.sub(r"[、。，．,.？?！!\s]+$", "", body)
    if tail and tail.endswith(TRAILING_PARTICLES):
        flags.append("文末が途切れ（助詞・接続で終わる）")
    elif body and body[-1] in "、，," and len(tail) > 3:
        flags.append("文末が読点で終わる")
    return flags


def main() -> int:
    csvs = sorted(ALIGNED_DIR.glob("*_aligned.csv"))
    if not csvs:
        print(f"{ALIGNED_DIR} に *_aligned.csv がありません（先に align_questions.py を実行）", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(exist_ok=True)
    summary = []

    for csv_path in csvs:
        base = csv_path.name[: -len("_aligned.csv")]
        if (TARGETS and base not in TARGETS) or base in EXCLUDE:
            continue
        rows = read_csv(csv_path)

        warn: dict[str, str] = {}
        tp = TRANS_DIR / f"{base}_transcript.csv"
        if tp.exists():
            warn = {r["ファイル名"]: r.get("要確認", "") for r in read_csv(tp)}
        conf: dict[str, list[float]] = {}
        wp = TRANS_DIR / f"{base}_words.csv"
        if wp.exists():
            for r in read_csv(wp):
                try:
                    conf.setdefault(r["ファイル名"], []).append(float(r["確信度"]))
                except (KeyError, ValueError):
                    pass
        ov: dict[str, float] = {}
        op = OVERLAP_DIR / base / "utterance_overlap.csv"
        if op.exists():
            for r in read_csv(op):
                try:
                    ov[r["ファイル名"]] = float(r["重なり率"])
                except (KeyError, ValueError):
                    pass

        out, n_flag = [], 0
        mismatches = []
        for r in rows:
            fn = r["ファイル名"]
            whisper, azure = r.get("文字起こし(Whisper)", ""), r.get("発言内容(元)", "")
            mm = mismatch(whisper, azure) if whisper and azure else None
            if mm is not None:
                mismatches.append(mm)
            cs = conf.get(fn, [])
            c_mean = sum(cs) / len(cs) if cs else None
            c_min = min(cs) if cs else None
            ovr = ov.get(fn)
            odd = oddness(r.get("発言内容(修正後)", ""), r.get("役割", ""))

            reasons = []
            if mm is not None and mm >= MISMATCH_FLAG:
                reasons.append("食い違い大")
            if c_mean is not None and c_mean < CONF_MEAN_FLAG:
                reasons.append("低確信")
            elif CONF_MIN_FLAG is not None and c_min is not None and c_min < CONF_MIN_FLAG:
                reasons.append("低確信(単語)")
            if warn.get(fn):
                reasons.append("Whisper警告")
            if ovr is not None and ovr >= OVERLAP_FLAG:
                reasons.append("重なり")
            if odd:
                reasons.append("文が不自然")
            n_flag += bool(reasons)
            out.append({
                "通し番号": r["通し番号"], "話者": r["話者"], "役割": r.get("役割", ""),
                "開始": r["開始"], "終了": r["終了"], "ファイル名": fn,
                "発言内容(修正後)": r.get("発言内容(修正後)", ""),
                "食い違い度": "" if mm is None else f"{mm:.2f}",
                "単語確信度(平均)": "" if c_mean is None else f"{c_mean:.2f}",
                "単語確信度(最小)": "" if c_min is None else f"{c_min:.2f}",
                "Whisperの警告": warn.get(fn, ""),
                "重なり率": "" if ovr is None else f"{ovr:.2f}",
                "文の不自然さ": " / ".join(odd),
                "要確認の理由": " ".join(reasons),
                "要確認の数": len(reasons),
            })
        cols = list(out[0].keys()) if out else []
        if out:
            write_csv(OUT_DIR / f"{base}_quality.csv", out, cols)
        summary.append({
            "mp3名": base, "発話数": len(rows), "要確認あり": n_flag,
            "要確認の割合": f"{n_flag / len(rows):.0%}" if rows else "",
            "食い違い度(平均)": f"{statistics.mean(mismatches):.2f}" if mismatches else "",
            "食い違い度(中央値)": f"{statistics.median(mismatches):.2f}" if mismatches else "",
            "食い違い大の発話": sum(1 for m in mismatches if m >= MISMATCH_FLAG),
            "文が不自然な発話": sum(1 for o in out if o["文の不自然さ"]),
            "重なりあり(率>=%d%%)" % round(OVERLAP_FLAG * 100): sum(1 for o in out if "重なり" in o["要確認の理由"]),
        })
        print(f"[{base}] 発話{len(rows)}件 / 要確認あり {n_flag}件 / 食い違い度の平均 "
              f"{(statistics.mean(mismatches) if mismatches else float('nan')):.2f}")

    if summary:
        write_csv(OUT_DIR / "quality_summary.csv", summary, list(summary[0].keys()))
    print(f"\n出力先: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())