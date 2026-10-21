#!/usr/bin/env python3
"""
発話の切れ目を、文字起こしの結果をもとに直して、元のmp3から切り直す（PyCharm用）。

Azureのタイムコードは秒単位で粗いので、切れ目がずれて、
    ・回答の音声の先頭に、前の質問の末尾（「ですか。」）が混ざる
    ・発話の末尾に、次の発話の頭が混ざる
    ・単語の途中で切れる
ことがあります。そこで次の手順を、切れ目が正しそうになるまで繰り返します。

    1. 元のmp3から、現在の切れ目で発話ごとに切り出す
    2. Whisperで文字起こしして、単語ごとの時刻を得る
    3. 切れ目の問題を検出する
        ・先頭の言葉が、前の発話（Azureの文字）の末尾と一致する → 前の発話の続きが混ざっている
        ・末尾の言葉が、次の発話（Azureの文字）の先頭と一致する → 次の発話の頭が混ざっている
        ・先頭／末尾の単語が、音声の端で切れている（確信度が低い）
    4. 混ざった言葉と本来の言葉の間（単語と単語の間の無音）に切れ目を移し、隣の発話の端も合わせて動かす
    5. 問題のない発話の割合が TARGET_OK_RATE に達するか、MAX_ITER 回まで繰り返す

前提: split_audio.py が済んでいる（csv_merged/*_merged.csv）。元のmp3が mp3/<mp3名>.mp3 にある。
      役割は out_roles/<mp3名>_roles.csv があればそれを使い、なければ推定する。

出力:
    out_refine/<mp3名>/refine_log.csv  各回の問題の一覧と、切れ目の変更
    out_refine/refine_summary.csv      ファイルごとの結果（各回の「問題なし」の割合）
    APPLY=True のとき:
        csv_merged/<mp3名>_merged.csv を、直した時刻（小数秒）で更新（元は csv_merged_original/ に退避）
        out_mp3/<mp3名>/ の、切れ目が変わった発話のmp3を、切り直したものに置き換え

※ 実行後は、assign_roles.py → transcribe.py → align_questions.py を、更新後のCSVで
   やり直してください（out_transcript/ などの古い出力は消してから）。

使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import difflib
import re
import shutil
import subprocess
import sys
from pathlib import Path

from align_questions import normalize
from assign_roles import infer_respondent, parse_time, read_csv, spk_num

# ============== 【設定】 ==============
TARGETS = []              # 処理する mp3 名（拡張子なし）。[] なら csv_merged 内のすべて
MAX_ITER = 5              # 切り直しの最大回数
TARGET_OK_RATE = 0.95     # 「切れ目に問題なし」の発話がこの割合に達したら終了
MIN_LEAK_CHARS = 3        # 混ざった言葉と判定する最小の文字数（ひらがな換算）
LEAK_SIMILARITY = 0.85    # 隣の発話の文字との一致度（0〜1）がこれ以上なら「混ざっている」
ADJ_GAP_SEC = 1.5         # 隣の発話との隙間がこの秒数以内なら「隣り合っている」とみなす
EDGE_SEC = 0.05           # 先頭／末尾の単語がこの秒数以内で端に接していて…
EDGE_PROB = 0.60          # …確信度がこの値未満なら「端で切れている疑い」
EDGE_STEP = 0.25          # 端で切れているときに、切れ目を広げる秒数
MAX_SHIFT_SEC = 5.0       # 1回に動かす切れ目の最大の幅
MIN_DUR_SEC = 0.30        # これより短くなる変更はしない
APPLY = True              # True: 最後に csv_merged と out_mp3 を更新する（元は退避）
# =====================================

BASE = Path(__file__).resolve().parent
MP3_DIR = BASE / "mp3"
MERGED_DIR = BASE / "csv_merged"
BACKUP_DIR = BASE / "csv_merged_original"
SPLIT_DIR = BASE / "out_mp3"
ROLES_DIR = BASE / "out_roles"
WORK_DIR = BASE / "out_refine"


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    return f"{int(sec) // 3600:02d}:{int(sec) % 3600 // 60:02d}:{sec % 60:05.2f}"


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def run_ffmpeg(args: list[str]) -> None:
    res = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip())


def cut_wav(src: Path, dst: Path, start: float, end: float) -> None:
    run_ffmpeg(["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)])


def cut_mp3(src: Path, dst: Path, start: float, end: float) -> None:
    run_ffmpeg(["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
                "-c:a", "libmp3lame", "-q:a", "2", str(dst)])


# --- 切れ目の問題の検出 ---
def _norm(s: str) -> str:
    return normalize(s)


def lead_leak(words: list[dict], prev_text: str) -> int:
    """先頭から何語が、前の発話の末尾と一致するか（0なら混ざりなし）"""
    prev = _norm(prev_text)
    best = 0
    for k in range(1, len(words)):  # 全部が一致する（何も残らない）場合は対象外
        a = _norm("".join(w["word"] for w in words[:k]))
        if len(a) < MIN_LEAK_CHARS:
            continue
        if len(a) > len(prev) or len(a) > 40:
            break
        if difflib.SequenceMatcher(None, a, prev[-len(a):]).ratio() >= LEAK_SIMILARITY:
            best = k
    return best


def tail_leak(words: list[dict], next_text: str) -> int:
    """末尾から何語が、次の発話の先頭と一致するか"""
    nxt = _norm(next_text)
    best = 0
    for k in range(1, len(words)):
        a = _norm("".join(w["word"] for w in words[-k:]))
        if len(a) < MIN_LEAK_CHARS:
            continue
        if len(a) > len(nxt) or len(a) > 40:
            break
        if difflib.SequenceMatcher(None, a, nxt[:len(a)]).ratio() >= LEAK_SIMILARITY:
            best = k
    return best


def evaluate(items: list[dict]) -> list[dict]:
    """各発話の切れ目の問題を調べる。items は時刻順。"""
    out = []
    for i, it in enumerate(items):
        words = it["words"]
        dur = it["end"] - it["start"]
        prev_text = items[i - 1]["text_azure"] if i > 0 else ""
        next_text = items[i + 1]["text_azure"] if i + 1 < len(items) else ""
        prob = {
            "lead": lead_leak(words, prev_text) if prev_text and words else 0,
            "tail": tail_leak(words, next_text) if next_text and words else 0,
            "edge_start": bool(words) and words[0]["start"] < EDGE_SEC and words[0]["prob"] < EDGE_PROB,
            "edge_end": bool(words) and words[-1]["end"] > dur - EDGE_SEC and words[-1]["prob"] < EDGE_PROB,
        }
        names = []
        if prob["lead"]:
            names.append("先頭に前の発話の続きが混ざっている")
        if prob["tail"]:
            names.append("末尾に次の発話の頭が混ざっている")
        if prob["edge_start"]:
            names.append("先頭の単語が切れている疑い")
        if prob["edge_end"]:
            names.append("末尾の単語が切れている疑い")
        prob["names"] = names
        prob["ok"] = not names
        out.append(prob)
    return out


def adjacent(a_end: float, b_start: float) -> bool:
    return b_start - a_end <= ADJ_GAP_SEC


def apply_fixes(items: list[dict], ev: list[dict]) -> set[int]:
    """問題のある切れ目を動かす。変わった発話の番号（items内の位置）を返す。"""
    changed: set[int] = set()

    def set_start(i: int, v: float) -> None:
        it = items[i]
        v = max(0.0, min(max(v, it["start"] - MAX_SHIFT_SEC), it["start"] + MAX_SHIFT_SEC))
        if it["end"] - v >= MIN_DUR_SEC and abs(v - it["start"]) > 0.01:
            it["start"] = v
            changed.add(i)

    def set_end(i: int, v: float) -> None:
        it = items[i]
        v = min(max(v, it["end"] - MAX_SHIFT_SEC), it["end"] + MAX_SHIFT_SEC)
        if v - it["start"] >= MIN_DUR_SEC and abs(v - it["end"]) > 0.01:
            it["end"] = v
            changed.add(i)

    for i, (it, pr) in enumerate(zip(items, ev)):
        words, t0 = it["words"], it["start"]
        has_prev = i > 0 and adjacent(items[i - 1]["end"], it["start"])
        has_next = i + 1 < len(items) and adjacent(it["end"], items[i + 1]["start"])
        if pr["lead"]:
            k = pr["lead"]
            cut = t0 + (words[k - 1]["end"] + words[k]["start"]) / 2
            set_start(i, cut)
            if has_prev:
                set_end(i - 1, max(items[i - 1]["end"], cut))  # 前の発話に、混ざっていた続きを戻す
        if pr["tail"]:
            k = pr["tail"]
            cut = t0 + (words[-k - 1]["end"] + words[-k]["start"]) / 2
            set_end(i, cut)
            if has_next:
                set_start(i + 1, min(items[i + 1]["start"], cut))  # 次の発話に、混ざっていた頭を戻す
        if pr["edge_start"] and not pr["lead"]:
            room = it["start"] - items[i - 1]["end"] if has_prev else EDGE_STEP
            set_start(i, it["start"] - min(EDGE_STEP, max(0.0, room)))
        if pr["edge_end"] and not pr["tail"]:
            room = items[i + 1]["start"] - it["end"] if has_next else EDGE_STEP
            set_end(i, it["end"] + min(EDGE_STEP, max(0.0, room)))
    return changed


def transcribe_cut(model, use_cuda, wav: Path, is_resp: bool, transcribe_mod) -> tuple[str, list[dict]]:
    res = transcribe_mod.transcribe_one(model, use_cuda, wav, is_resp)
    text, segs, _ = transcribe_mod.filter_result(res)
    words = [{"word": w["word"].strip(), "start": float(w["start"]), "end": float(w["end"]),
              "prob": float(w.get("probability", 1.0))}
             for s in segs for w in s.get("words", []) if w["word"].strip()]
    return text, words


def refine_file(base: str, rows: list[dict], is_resp_of: dict[str, bool], src: Path,
                transcribe_fn) -> tuple[list[dict], list[dict], list[dict]]:
    """切れ目を直す。(直した発話の一覧, ログ, 各回の集計) を返す。
    transcribe_fn(wav_path, is_resp) -> (本文, 単語リスト)"""
    work = WORK_DIR / base
    work.mkdir(parents=True, exist_ok=True)
    items = []
    for r in rows:
        items.append({"row": r, "start": parse_time(r["開始"]), "end": parse_time(r["終了"]),
                      "orig_start": parse_time(r["開始"]), "orig_end": parse_time(r["終了"]),
                      "text_azure": r["発話内容"], "words": [], "text": ""})
    items.sort(key=lambda x: x["start"])
    log, iters = [], []

    def do_transcribe(idx: int) -> None:
        it = items[idx]
        wav = work / f"{Path(it['row']['ファイル名']).stem}.wav"
        cut_wav(src, wav, it["start"], it["end"])
        it["text"], it["words"] = transcribe_fn(wav, is_resp_of.get(it["row"]["ファイル名"], True))

    for idx in range(len(items)):
        do_transcribe(idx)
        if (idx + 1) % 20 == 0 or idx + 1 == len(items):
            print(f"  [{base}] 初回の文字起こし {idx + 1}/{len(items)}")

    for it_no in range(MAX_ITER + 1):
        ev = evaluate(items)
        n_ok = sum(1 for e in ev if e["ok"])
        rate = n_ok / len(items) if items else 1.0
        for it, e in zip(items, ev):
            if not e["ok"]:
                log.append({"回": it_no, "通し番号": it["row"]["通し番号"], "ファイル名": it["row"]["ファイル名"],
                            "問題": " / ".join(e["names"]), "開始": f"{it['start']:.2f}", "終了": f"{it['end']:.2f}",
                            "先頭の本文": it["text"][:30]})
        iters.append({"mp3名": base, "回": it_no, "発話数": len(items), "問題なし": n_ok, "問題なしの割合": f"{rate:.1%}"})
        print(f"  [{base}] {it_no}回目: 問題なし {n_ok}/{len(items)}（{rate:.1%}）")
        if rate >= TARGET_OK_RATE or it_no == MAX_ITER:
            break
        changed = apply_fixes(items, ev)
        if not changed:
            print(f"  [{base}] これ以上、動かせる切れ目がありません")
            break
        print(f"  [{base}] 切れ目を変更: {len(changed)}件 → 切り直して文字起こし")
        for idx in sorted(changed):
            do_transcribe(idx)
    return items, log, iters


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg が見つかりません。", file=sys.stderr)
        return 1
    csvs = sorted(MERGED_DIR.glob("*_merged.csv"))
    if not csvs:
        print(f"{MERGED_DIR} にマージ済みCSVがありません（先に split_audio.py を実行）", file=sys.stderr)
        return 1
    import transcribe as tm

    WORK_DIR.mkdir(exist_ok=True)
    model, use_cuda = None, False
    summary = []

    for csv_path in csvs:
        rows = read_csv(csv_path)
        if not rows:
            continue
        base = re.sub(r"_\d+_spk\d+\.mp3$", "", rows[0]["ファイル名"])
        if TARGETS and base not in TARGETS:
            continue
        src = MP3_DIR / f"{base}.mp3"
        if not src.exists():
            print(f"[{base}] 元のmp3がありません（{src.name}）。スキップします")
            summary.append({"mp3名": base, "結果": "元のmp3なし"})
            continue
        if model is None:
            model, use_cuda = tm.load_model()

        roles_p = ROLES_DIR / f"{base}_roles.csv"
        if roles_p.exists():
            is_resp_of = {r["ファイル名"]: r["役割"] == "回答者" for r in read_csv(roles_p)}
        else:
            resp = infer_respondent(rows)[0]
            is_resp_of = {r["ファイル名"]: spk_num(r["話者"]) == resp for r in rows}

        print(f"[{base}] 切れ目の修正を開始（{len(rows)}発話）")
        items, log, iters = refine_file(
            base, rows, is_resp_of, src,
            lambda wav, is_resp: transcribe_cut(model, use_cuda, wav, is_resp, tm))
        write_csv(WORK_DIR / base / "refine_log.csv", log,
                  ["回", "通し番号", "ファイル名", "問題", "開始", "終了", "先頭の本文"])
        last = iters[-1]
        summary.append({"mp3名": base, "結果": "完了", "回数": last["回"], "最終の問題なしの割合": last["問題なしの割合"],
                        "最初の問題なしの割合": iters[0]["問題なしの割合"],
                        "切れ目が変わった発話": sum(1 for it in items
                                          if abs(it["start"] - it["orig_start"]) > 0.01 or abs(it["end"] - it["orig_end"]) > 0.01)})

        if APPLY:
            BACKUP_DIR.mkdir(exist_ok=True)
            bk = BACKUP_DIR / csv_path.name
            if not bk.exists():
                shutil.copy(csv_path, bk)
            new_rows, n_cut = [], 0
            out_dir = SPLIT_DIR / base
            out_dir.mkdir(parents=True, exist_ok=True)
            for it in items:
                r = dict(it["row"])
                r["開始"], r["終了"] = fmt_time(it["start"]), fmt_time(it["end"])
                new_rows.append(r)
                moved = abs(it["start"] - it["orig_start"]) > 0.01 or abs(it["end"] - it["orig_end"]) > 0.01
                dst = out_dir / r["ファイル名"]
                if moved or not dst.exists():
                    cut_mp3(src, dst, it["start"], it["end"])
                    n_cut += 1
            new_rows.sort(key=lambda r: int(r["通し番号"]))
            write_csv(csv_path, new_rows, list(rows[0].keys()))
            print(f"[{base}] csv_merged を更新、mp3を{n_cut}件切り直しました（元のCSVは {BACKUP_DIR.name}/ に退避）")

    if summary:
        cols = []
        for s in summary:
            for k in s:
                if k not in cols:
                    cols.append(k)
        write_csv(WORK_DIR / "refine_summary.csv", summary, cols)
    print(f"\n出力先: {WORK_DIR}\n（このあと assign_roles.py → transcribe.py → align_questions.py をやり直してください）")
    return 0


if __name__ == "__main__":
    sys.exit(main())