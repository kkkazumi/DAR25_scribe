#!/usr/bin/env python3
"""
各ファイルで「質問者／回答者」を推定し、名前（「○○さん」）を伏せ字にする（PyCharm用）。

前提: split_audio.py が済んでいて、csv_merged/*_merged.csv ができている

処理:
    1. 発言内容から、質問者／回答者を推定（質問で終わる発話の割合が高い方＝質問者。差が小さければ発話時間）
    2. 全発話に「役割」を付け、発言内容の「○○さん」のような名前を伏せ字にする（regexベース）

出力（out_roles/）:
    <mp3名>_roles.csv       全発話（質問者・回答者）＋役割＋名前を伏せた発言内容
    respondent_decision.csv 各ファイルの質問者／回答者の判定結果（要確認）
    name_mask_log.csv       伏せ字にした語の一覧（要確認）

使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import re
import sys
from pathlib import Path

# ============== 【設定】 ==============
TARGETS = []                 # 処理する mp3 名（拡張子なし）。[] なら csv_merged 内のすべて
RESPONDENT_OVERRIDE = {}     # 推定を使わず指定する。例: {"DAR25_001_Human": 2}
DEFAULT_RESPONDENT = 2       # 推定できないときの回答者（話者番号）
MASK_TEXT = "○○"            # 名前の置換後の文字
# =====================================

BASE = Path(__file__).resolve().parent
MERGED_DIR = BASE / "csv_merged"
OUT_DIR = BASE / "out_roles"

QUESTION_END = re.compile(r"(か|[？?])[。、．\s]*$")

HONORIFICS = "さん|くん|君|ちゃん|様|氏|先生"
NAME_RE = re.compile(r"([一-龥々]{1,4}|[ァ-ヶー]{2,8})(" + HONORIFICS + ")")
# 名前ではない語（「お母さん」「ヘルパーさん」など）は伏せない
NOT_NAMES = {
    "皆", "母", "父", "客", "奥", "旦那", "患者", "娘", "息子", "兄", "姉", "弟", "妹", "孫",
    "子供", "叔父", "叔母", "伯父", "伯母", "医者", "看護師", "先輩", "後輩", "友達", "夫", "妻",
    "主人", "家族", "神", "赤", "お", "祖父", "祖母", "親", "子", "他", "皆様", "お客",
    "ヘルパー", "ケアマネ", "ケアマネージャー", "スタッフ", "ボランティア", "ナース", "ドクター",
    "ママ", "パパ", "ロボット", "ペッパー", "ペッパ", "ポッド",
}


def mask_names(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    def repl(m: re.Match) -> str:
        if m.group(1) in NOT_NAMES:
            return m.group(0)
        found.append(m.group(0))
        return MASK_TEXT + m.group(2)

    return NAME_RE.sub(repl, text), found


def parse_time(s: str) -> float:
    sec = 0.0
    for part in s.strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def spk_num(label: str) -> int:
    return int(re.search(r"\d+", label).group())


def infer_respondent(rows: list[dict]) -> tuple[int, str, dict]:
    stats: dict[int, dict] = {}
    for r in rows:
        s = stats.setdefault(spk_num(r["話者"]), {"turns": 0, "q": 0, "sec": 0.0})
        s["turns"] += 1
        s["sec"] += parse_time(r["終了"]) - parse_time(r["開始"])
        if QUESTION_END.search(r["発話内容"]):
            s["q"] += 1
    if len(stats) != 2:
        return DEFAULT_RESPONDENT, f"話者が{len(stats)}人のため既定値", stats

    (a, sa), (b, sb) = sorted(stats.items())
    ra, rb = sa["q"] / sa["turns"], sb["q"] / sb["turns"]
    if abs(ra - rb) >= 0.10:  # 質問で終わる発話の割合の差が十分ある
        interviewer = a if ra > rb else b
        return (b if interviewer == a else a), "質問率", stats
    if abs(sa["sec"] - sb["sec"]) >= 0.10 * (sa["sec"] + sb["sec"]):  # 発話時間の差
        interviewer = a if sa["sec"] < sb["sec"] else b
        return (b if interviewer == a else a), "発話時間", stats
    return DEFAULT_RESPONDENT, "判定不能のため既定値", stats


def main() -> int:
    csvs = sorted(MERGED_DIR.glob("*_merged.csv"))
    if not csvs:
        print(f"マージ済みCSVが見つかりません: {MERGED_DIR}（先に split_audio.py を実行）", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(exist_ok=True)
    decisions, mask_log = [], []

    for csv_path in csvs:
        rows = read_csv(csv_path)
        if not rows:
            continue
        base = re.sub(r"_\d+_spk\d+\.mp3$", "", rows[0]["ファイル名"])
        if TARGETS and base not in TARGETS:
            continue
        if base in RESPONDENT_OVERRIDE:
            resp, method, stats = RESPONDENT_OVERRIDE[base], "手動指定", {}
        else:
            resp, method, stats = infer_respondent(rows)
        decisions.append({"mp3名": base, "回答者": f"話者{resp}", "判定方法": method,
                          "統計": "; ".join(f"話者{k}: 質問率{v['q']}/{v['turns']}・{v['sec']:.0f}秒"
                                          for k, v in sorted(stats.items()))})
        out_rows, n_masked = [], 0
        for r in rows:
            masked, found = mask_names(r["発話内容"])
            n_masked += len(found)
            for w in found:
                mask_log.append({"mp3名": base, "通し番号": r["通し番号"], "元の語": w, "文脈": r["発話内容"][:60]})
            out_rows.append({
                "通し番号": r["通し番号"], "話者": r["話者"],
                "役割": "回答者" if spk_num(r["話者"]) == resp else "質問者",
                "開始": r["開始"], "終了": r["終了"], "ファイル名": r["ファイル名"], "発話内容": masked,
            })
        write_csv(OUT_DIR / f"{base}_roles.csv", out_rows,
                  ["通し番号", "話者", "役割", "開始", "終了", "ファイル名", "発話内容"])
        print(f"[{base}] 回答者=話者{resp}（{method}） 発話{len(out_rows)}件 / 名前の伏せ字 {n_masked}件")

    write_csv(OUT_DIR / "respondent_decision.csv", decisions, ["mp3名", "回答者", "判定方法", "統計"])
    if mask_log:
        write_csv(OUT_DIR / "name_mask_log.csv", mask_log, ["mp3名", "通し番号", "元の語", "文脈"])
    print(f"\n出力先: {OUT_DIR}\n（respondent_decision.csv と name_mask_log.csv は、必ず目を通してください）")
    return 0


if __name__ == "__main__":
    sys.exit(main())