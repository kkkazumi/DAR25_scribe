#!/usr/bin/env python3
"""
外注の振り分け表を、Human条件とRobot条件で分けて作る（PyCharm用）。

前提: check_quality.py（out_quality/*_quality.csv）と refine_boundaries.py（out_refine/refine_summary.csv）まで済んでいる。

【Human条件】 録音全体（質問者・回答者の両方）の文字起こしが必要
    ・話者が1人だけの面談           → コエラボ：素起こし＋タイムスタンプ
    ・精度が高く、誤検出が少ない面談 → データグリーン：文字起こし結果の校正（タイムスタンプは無視）
    ・それ以外                       → コエラボ：素起こしのみ（タイムスタンプは無視）
    ※ 指標は、質問者・回答者の全発話で計算する

【Robot条件】 ロボットの発言は、ログで分かっているので、回答者の発言だけが対象
    ・回答者の発言を切り出して連結した音声を、データグリーンに校正依頼する
    ・回答者の発話だけで指標を計算し、基準を超えるものは「要注意」とし、コエラボ（素起こしのみ）に出す
      （R_WARN_TO_KOELABO = False にすると、データグリーンのまま「要注意」を付けるだけになる）
    ・ログと録音が合わない面談（025 など）は、備考に出す

指標（面談ごと）:
    精度(一致度)           = 1 − 食い違い度の平均（Whisperと元の文字起こしがどれだけ一致するか）
    食い違い大の割合       = 食い違い度が STRONG_MISMATCH 以上の発話の割合（誤検出の目安）
    切れ目が合っている割合 = refine_boundaries.py の「最終の問題なしの割合」
    重なり多の割合         = 2人が同時に話している割合が高い発話の割合
    文の不自然さの割合     = 途中で切れていそう・前の発話の末尾が混ざっていそう、などの割合

出力: out_triage/triage_Human.csv, triage_Robot.csv, 図 triage_Human.png / triage_Robot.png（matplotlibがあれば）
使い方: 下の【設定】のしきい値を、分布を見ながら調整して、PyCharm で右クリック → Run
"""
import csv
import sys
from pathlib import Path

# ============== 【設定】 ==============
SINGLE_SPEAKER = ["DAR25_007_Human", "DAR25_008_Human", "DAR25_031_Human", "DAR25_047_Human"]
STRONG_MISMATCH = 0.30     # 食い違い度がこれ以上の発話を「食い違い大」とする

# --- Human: 「精度が高い」とみなす条件（すべて満たすとき）---
H_MAX_MISMATCH = 0.28      # 食い違い度の平均がこれ以下
H_MAX_BIG_RATE = 0.33      # 食い違い大の割合がこれ以下
H_MIN_BOUNDARY_OK = 0.60   # 切れ目が合っている割合がこれ以上
H_MAX_OVERLAP_RATE = 0.30  # 重なり多の割合がこれ以下

# --- Robot: 「要注意」とみなす条件（どれかに当てはまるとき）。回答者の発話だけで計算 ---
R_WARN_MISMATCH = 0.15     # 食い違い度の平均がこれ以上
R_WARN_BIG_RATE = 0.25     # 食い違い大の割合がこれ以上
R_WARN_BOUNDARY_OK = 0.85  # 切れ目が合っている割合がこれ未満
R_WARN_OVERLAP_RATE = 0.10 # 重なり多の割合がこれ以上
R_WARN_TO_KOELABO = True   # True: 要注意のRobot面談は、データグリーン（校正）ではなく、コエラボ（素起こしのみ）に出す
LOG_MISMATCH_FILES = ["DAR25_025_Robot"]   # ログと録音が合わなかった面談（備考に出す）
NO_LOG_FILES = ["DAR25_006_Robot", "DAR25_018_Robot"]  # ログがない面談
# =====================================

BASE = Path(__file__).resolve().parent
QUALITY_DIR = BASE / "out_quality"
REFINE = BASE / "out_refine" / "refine_summary.csv"
OUT = BASE / "out_triage"


def read(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pct(s: str) -> float:
    return float(str(s).replace("%", "")) / 100 if s not in ("", None) else 0.0


def metrics(rows: list[dict]) -> dict | None:
    mm = [float(r["食い違い度"]) for r in rows if r.get("食い違い度") not in ("", None)]
    if not rows or not mm:
        return None
    n = len(rows)
    return {"発話数": n,
            "mm": sum(mm) / len(mm),
            "big": sum(1 for x in mm if x >= STRONG_MISMATCH) / len(mm),
            "ov": sum(1 for r in rows if "重なり" in r.get("要確認の理由", "")) / n,
            "unn": sum(1 for r in rows if r.get("文の不自然さ")) / n}


def write(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict], colors: dict, title: str, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = ["MS Gothic", "Yu Gothic", "Meiryo", "IPAexGothic", "sans-serif"]
        fig, ax = plt.subplots(figsize=(8, 6))
        for dest, c in colors.items():
            pts = [r for r in rows if r["依頼先(案)"] == dest and r["精度(一致度)"]]
            ax.scatter([float(r["精度(一致度)"]) for r in pts], [float(r["切れ目が合っている割合"]) for r in pts],
                       c=c, label=f"{dest}（{len(pts)}）")
            for r in pts:
                ax.annotate(r["mp3名"][6:9], (float(r["精度(一致度)"]), float(r["切れ目が合っている割合"])), fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("精度（一致度）= 1 − 食い違い度")
        ax.set_ylabel("切れ目が合っている割合")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[図は作れませんでした] {e}")


def main() -> int:
    if not REFINE.exists() or not QUALITY_DIR.exists():
        print("out_quality と out_refine/refine_summary.csv が必要です", file=sys.stderr)
        return 1
    refine = {r["mp3名"]: r for r in read(REFINE)}
    OUT.mkdir(exist_ok=True)
    human, robot = [], []
    for qp in sorted(QUALITY_DIR.glob("*_quality.csv")):
        name = qp.name[:-len("_quality.csv")]
        rows = read(qp)
        ok = pct(refine.get(name, {}).get("最終の問題なしの割合", "0"))
        if name.endswith("_Human"):
            m = metrics(rows)
            if m is None:
                continue
            bad = []
            if m["mm"] > H_MAX_MISMATCH: bad.append(f"食い違い度{m['mm']:.2f}")
            if m["big"] > H_MAX_BIG_RATE: bad.append(f"食い違い大{m['big']:.0%}")
            if ok < H_MIN_BOUNDARY_OK: bad.append(f"切れ目OK{ok:.0%}")
            if m["ov"] > H_MAX_OVERLAP_RATE: bad.append(f"重なり{m['ov']:.0%}")
            dest = "コエラボ：素起こしのみ" if bad else "データグリーン：校正"
            human.append({"mp3名": name, "発話数": m["発話数"], "食い違い度(平均)": f"{m['mm']:.2f}", "精度(一致度)": f"{1 - m['mm']:.2f}",
                          "食い違い大の割合": f"{m['big']:.2f}", "切れ目が合っている割合": f"{ok:.2f}",
                          "重なり多の割合": f"{m['ov']:.2f}", "文の不自然さの割合": f"{m['unn']:.2f}",
                          "依頼先(案)": dest, "理由": " / ".join(bad) if bad else "基準を満たす"})
        elif name.endswith("_Robot"):
            m = metrics([r for r in rows if r.get("役割") == "回答者"])
            if m is None:
                continue
            warn = []
            if m["mm"] >= R_WARN_MISMATCH: warn.append(f"食い違い度{m['mm']:.2f}")
            if m["big"] >= R_WARN_BIG_RATE: warn.append(f"食い違い大{m['big']:.0%}")
            if ok < R_WARN_BOUNDARY_OK: warn.append(f"切れ目OK{ok:.0%}")
            if m["ov"] >= R_WARN_OVERLAP_RATE: warn.append(f"重なり{m['ov']:.0%}")
            note = []
            if name in LOG_MISMATCH_FILES: note.append("ログと録音が合わない（台本との突き合わせで処理）")
            if name in NO_LOG_FILES: note.append("ログなし（台本との突き合わせで処理）")
            robot.append({"mp3名": name, "回答者の発話数": m["発話数"], "精度(一致度)": f"{1 - m['mm']:.2f}",
                          "食い違い大の割合": f"{m['big']:.2f}", "切れ目が合っている割合": f"{ok:.2f}",
                          "重なり多の割合": f"{m['ov']:.2f}", "文の不自然さの割合": f"{m['unn']:.2f}",
                          "依頼先(案)": (("コエラボ：素起こしのみ" if R_WARN_TO_KOELABO else "データグリーン：校正（要注意）")
                                      if warn else "データグリーン：校正"),
                          "理由": " / ".join(warn), "備考": " ".join(note)})
    for name in SINGLE_SPEAKER:
        human.append({"mp3名": name, "発話数": "", "食い違い度(平均)": "", "精度(一致度)": "", "食い違い大の割合": "", "切れ目が合っている割合": "",
                      "重なり多の割合": "", "文の不自然さの割合": "", "依頼先(案)": "コエラボ：素起こし＋タイムスタンプ",
                      "理由": "話者が1人だけ"})
    human = [h for h in human if not (h["理由"] != "話者が1人だけ" and h["mp3名"] in SINGLE_SPEAKER)]
    human.sort(key=lambda r: (r["依頼先(案)"], -float(r["精度(一致度)"] or 0)))
    robot.sort(key=lambda r: (r["依頼先(案)"], -float(r["精度(一致度)"] or 0)))
    for label, rows in (("Human", human), ("Robot", robot)):
        if rows:
            write(OUT / f"triage_{label}.csv", rows)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["依頼先(案)"]] = counts.get(r["依頼先(案)"], 0) + 1
        print(f"{label}条件の振り分け（案）:")
        for k, v in sorted(counts.items()):
            print(f"  {k}: {v}件")
    dg = [{"mp3名": h["mp3名"], "発話数": h["発話数"], "食い違い度(平均)": h["食い違い度(平均)"],
           "食い違い大の割合": h["食い違い大の割合"], "切れ目が合っている割合": h["切れ目が合っている割合"],
           "重なり多の割合": h["重なり多の割合"]} for h in human if h["依頼先(案)"].startswith("データグリーン")]
    if dg:
        dg.sort(key=lambda r: float(r["食い違い度(平均)"]))
        write(OUT / "triage_Human_データグリーン.csv", dg)
        print(f"\nHumanでデータグリーン行きの4指標: triage_Human_データグリーン.csv（{len(dg)}件）")
    plot(human, {"データグリーン：校正": "tab:blue", "コエラボ：素起こしのみ": "tab:orange"},
         "Human条件（全発話）", OUT / "triage_Human.png")
    plot(robot, {"データグリーン：校正": "tab:blue", "データグリーン：校正（要注意）": "tab:red", "コエラボ：素起こしのみ": "tab:orange"},
         "Robot条件（回答者の発話のみ）", OUT / "triage_Robot.png")
    print(f"\n出力先: {OUT}（triage_Human.csv / triage_Robot.csv を見て、しきい値を調整してください）")
    return 0


if __name__ == "__main__":
    sys.exit(main())