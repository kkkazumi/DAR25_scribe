#!/usr/bin/env python3
"""
暫定版の「きれいなCSV」を作る（PyCharm用）。

out_aligned/*_aligned.csv には、修正前・修正後、判定、類似度など、作業用の列がたくさん入っている。
そこから、タイミング・話者・発話内容など、使う列だけを取り出して、面談ごとのCSVと、全面談をまとめたCSVを作る。

列: 面談, 条件, 通し番号, 開始, 終了, 話者, 役割, 質問ID, 発話内容, 要確認(任意)
  ・発話内容 … 「発言内容(修正後)」。名前は「○○」にする（make_handover_list.py と同じ方法で追加の伏せ字も行う）
  ・Robotでログの文に置き換えたとき、1つの発話が複数行に分かれていた場合は、1行にまとめる（終了は最後の行の時刻）
  ・本文が空の行は、除く
  ・話者が1人しか分離できなかった面談（align_questions.py の EXCLUDE）は、out_aligned がないので含まれない

出力: out_clean/<mp3名>_clean.csv、out_clean/all_clean.csv
使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import sys
from pathlib import Path

from make_handover_list import after_name_flags, load_known, mask_text, name_spans

# ============== 【設定】 ==============
TARGETS = []               # 処理する mp3 名。[] なら out_aligned のすべて
INCLUDE_FLAG = True        # True: 「要確認」の列を付ける（自動処理で不安な発話の目印）
ONLY_RESPONDENT = False    # True: 回答者の発話だけにする
# =====================================

BASE = Path(__file__).resolve().parent
ALIGNED = BASE / "out_aligned"
OUT = BASE / "out_clean"


def read(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def clean(base: str, rows: list[dict], known: set[str]) -> list[dict]:
    flags = after_name_flags(rows)
    out: list[dict] = []
    for r, fl in zip(rows, flags):
        text = r.get("発言内容(修正後)", "").strip()
        if not text:
            # ログの文を最初の行にまとめたときの、残りの行。直前の発話の終了時刻を延ばす
            if out and r.get("判定", "").startswith("ロボットログ（前の行に含む）") and out[-1]["役割"] == r["役割"]:
                out[-1]["終了"] = r["終了"]
            continue
        text = mask_text(text, name_spans(text, known, fl))
        rec = {"面談": base, "条件": "Human" if base.endswith("_Human") else "Robot",
               "通し番号": r["通し番号"], "開始": r["開始"], "終了": r["終了"], "話者": r["話者"],
               "役割": r["役割"], "質問ID": r.get("質問ID", ""), "発話内容": text}
        if INCLUDE_FLAG:
            rec["要確認"] = r.get("要確認", "")
        out.append(rec)
    for i, rec in enumerate(out, start=1):
        rec["通し番号"] = i
    return out


def main() -> int:
    files = sorted(ALIGNED.glob("*_aligned.csv"))
    if not files:
        print(f"{ALIGNED} に *_aligned.csv がありません", file=sys.stderr)
        return 1
    OUT.mkdir(exist_ok=True)
    known_all = load_known()
    everything = []
    for p in files:
        base = p.name[:-len("_aligned.csv")]
        if TARGETS and base not in TARGETS:
            continue
        rows = clean(base, read(p), known_all.get(base, set()))
        if ONLY_RESPONDENT:
            rows = [r for r in rows if r["役割"] == "回答者"]
        if not rows:
            continue
        write(OUT / f"{base}_clean.csv", rows)
        everything.extend(rows)
        print(f"[{base}] {len(rows)}発話")
    if everything:
        write(OUT / "all_clean.csv", everything)
    print(f"\n全 {len(everything)} 発話 → {OUT / 'all_clean.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())