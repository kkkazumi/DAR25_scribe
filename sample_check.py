#!/usr/bin/env python3
"""
外注の振り分けが妥当か、ランダムに選んだ発話で確かめる（PyCharm用）。

前提: make_handover_list.py まで済んでいる（out_handover/ のリストがある）。

1回目の実行:
    ・依頼先の群（データグリーン Human / Robot、コエラボ Human / Robot）ごとに、回答者の発話を SAMPLE_PER_GROUP 件ずつ、
      ランダムに選ぶ（SEED で固定。同じ設定なら同じ発話が選ばれる）。
    ・out_sample/sample_check.csv … 選んだ発話（文字起こし、元の文字起こし、各指標、判定の空欄）
    ・out_sample/audio/            … 選んだ発話の分割音声（聞いて確かめる用）。※名前は無音にしていないので、外に出さないこと
    ・out_sample/group_stats.csv   … 群ごとの全発話の指標（食い違い度の平均、食い違い大の割合 など）
聞いて確かめる:
    ・sample_check.csv の「判定」に、○ / △ / × を入れる
        ○ = そのまま使える（誤りなし、または気にならない程度）
        △ = 軽い誤り（校正で直せる）
        × = 大きな誤り（聞き直しが必要）
    ・保存して、もう一度このプログラムを実行する（判定が入っていれば、選び直さずに集計する）
2回目の実行:
    ・out_sample/judge_summary.csv … 群ごとの ○ △ × の割合、判定ごとの食い違い度の平均
使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import random
import shutil
import sys
from pathlib import Path

# ============== 【設定】 ==============
SAMPLE_PER_GROUP = 10      # 群ごとに選ぶ発話の数
MIN_CHARS = 8              # これより短い発話（「はい」など）は選ばない
SEED = 2026                # 乱数の種（変えると、別の発話が選ばれる）
REGENERATE = False         # True: 判定が空のときでも、選び直す
STRONG_MISMATCH = 0.30     # 食い違い度がこれ以上を「食い違い大」とする
# =====================================

BASE = Path(__file__).resolve().parent
HANDOVER = BASE / "out_handover"
QUALITY = BASE / "out_quality"
ALIGNED = BASE / "out_aligned"
SPLIT = BASE / "out_mp3"
OUT = BASE / "out_sample"


def read(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def f(x: str) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def mean(xs: list[float]) -> str:
    return f"{sum(xs) / len(xs):.2f}" if xs else ""


def groups() -> dict[str, list[str]]:
    """群の名前 → mp3名の一覧"""
    g: dict[str, list[str]] = {}
    dg, kl = HANDOVER / "データグリーン_校正_リスト.csv", HANDOVER / "コエラボ_素起こし_リスト.csv"
    for p, label in ((dg, "データグリーン校正"), (kl, "コエラボ")):
        if not p.exists():
            continue
        for r in read(p):
            audio = r["渡す音声"]
            base = audio.replace("_回答者のみ.mp3", "").replace(".mp3", "")
            if r.get("依頼内容", "").startswith("素起こし＋"):
                continue  # 話者が1人の面談は、回答者の区別がないので対象外
            g.setdefault(f"{label}_{r['区分']}", []).append(base)
    return g


def respondent_rows(base: str) -> list[dict]:
    qp, ap = QUALITY / f"{base}_quality.csv", ALIGNED / f"{base}_aligned.csv"
    if not qp.exists() or not ap.exists():
        return []
    orig = {r["ファイル名"]: r.get("発言内容(元)", "") for r in read(ap)}
    rows = []
    for r in read(qp):
        if r.get("役割") != "回答者":
            continue
        r["_base"], r["_orig"] = base, orig.get(r["ファイル名"], "")
        rows.append(r)
    return rows


def stats(rows: list[dict]) -> dict:
    mm = [x for x in (f(r["食い違い度"]) for r in rows) if x is not None]
    cf = [x for x in (f(r["単語確信度(平均)"]) for r in rows) if x is not None]
    return {"発話数": len(rows),
            "食い違い度の平均": mean(mm),
            "食い違い大の割合": f"{sum(1 for x in mm if x >= STRONG_MISMATCH) / len(mm):.2f}" if mm else "",
            "単語確信度の平均": mean(cf),
            "重なりのある発話の割合": f"{sum(1 for r in rows if '重なり' in r.get('要確認の理由', '')) / len(rows):.2f}" if rows else "",
            "文が不自然な発話の割合": f"{sum(1 for r in rows if r.get('文の不自然さ')) / len(rows):.2f}" if rows else ""}


def summarize(path: Path) -> None:
    rows = read(path)
    out, by_judge = [], {}
    gs = sorted({r["群"] for r in rows})
    for g in gs:
        sub = [r for r in rows if r["群"] == g and r["判定"].strip()]
        if not sub:
            continue
        n = len(sub)
        c = {k: sum(1 for r in sub if r["判定"].strip() == k) for k in ("○", "△", "×")}
        out.append({"群": g, "判定した数": n, "○（そのまま使える）": f"{c['○']}（{c['○'] / n:.0%}）",
                    "△（軽い誤り）": f"{c['△']}（{c['△'] / n:.0%}）", "×（大きな誤り）": f"{c['×']}（{c['×'] / n:.0%}）"})
    for r in rows:
        j = r["判定"].strip()
        v = f(r["食い違い度"])
        if j and v is not None:
            by_judge.setdefault(j, []).append(v)
    if not out:
        print("判定の列が空です。聞いて、○ / △ / × を入れてください。")
        return
    write(OUT / "judge_summary.csv", out)
    print("群ごとの判定:")
    for r in out:
        print(f"  {r['群']}: ○ {r['○（そのまま使える）']} / △ {r['△（軽い誤り）']} / × {r['×（大きな誤り）']}（n={r['判定した数']}）")
    print("判定ごとの食い違い度の平均（指標が人の判断と合っているかの目安）:")
    for j in ("○", "△", "×"):
        if j in by_judge:
            print(f"  {j}: {mean(by_judge[j])}（n={len(by_judge[j])}）")
    print(f"\n出力先: {OUT / 'judge_summary.csv'}")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    sp = OUT / "sample_check.csv"
    if sp.exists() and not REGENERATE:
        rows = read(sp)
        if any(r["判定"].strip() for r in rows):
            summarize(sp)
            return 0
        print("sample_check.csv は作成済みで、判定が空です。聞いて ○ / △ / × を入れてから、もう一度実行してください。")
        print("（選び直すときは REGENERATE = True）")
        return 0
    g = groups()
    if not g:
        print("先に make_handover_list.py を実行してください", file=sys.stderr)
        return 1
    rng = random.Random(SEED)
    shutil.rmtree(OUT / "audio", ignore_errors=True)
    (OUT / "audio").mkdir(parents=True)
    sample, gstats = [], []
    for name in sorted(g):
        allrows = [r for b in g[name] for r in respondent_rows(b)]
        if not allrows:
            continue
        s = stats(allrows)
        gstats.append({"群": name, "面談数": len(g[name]), **{("回答者の" + k if k == "発話数" else k): v for k, v in s.items()}})
        pool = [r for r in allrows if len(r["発言内容(修正後)"]) >= MIN_CHARS]
        for k, r in enumerate(rng.sample(pool, min(SAMPLE_PER_GROUP, len(pool))), start=1):
            src = SPLIT / r["_base"] / r["ファイル名"]
            audio_name = f"{name}_{k:02d}_{r['_base']}_{r['通し番号']}.mp3"
            if src.exists():
                shutil.copy2(src, OUT / "audio" / audio_name)
            sample.append({"群": name, "番号": k, "mp3名": r["_base"], "通し番号": r["通し番号"],
                           "音声(out_sample/audio)": audio_name if src.exists() else "（分割音声なし）",
                           "文字起こし(修正後)": r["発言内容(修正後)"], "元の文字起こし(Azure)": r["_orig"],
                           "食い違い度": r["食い違い度"], "単語確信度(平均)": r["単語確信度(平均)"],
                           "重なり率": r["重なり率"], "判定": "", "直した文": "", "メモ": ""})
    write(sp, sample)
    write(OUT / "group_stats.csv", gstats)
    print("群ごとの全発話の指標:")
    for r in gstats:
        print(f"  {r['群']}: {r['面談数']}面談 / 食い違い度の平均 {r['食い違い度の平均']} / 食い違い大 {r['食い違い大の割合']}"
              f" / 単語確信度 {r['単語確信度の平均']} / 重なり {r['重なりのある発話の割合']}")
    print(f"\nランダムに {len(sample)} 発話を選びました: {sp}")
    print("out_sample/audio/ の音声を聞いて、sample_check.csv の『判定』に ○ / △ / × を入れ、もう一度実行してください。")
    return 0


if __name__ == "__main__":
    sys.exit(main())