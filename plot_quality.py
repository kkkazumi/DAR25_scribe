#!/usr/bin/env python3
"""
check_quality.py の結果（out_quality/*_quality.csv）から、指標の分布を図と表にする（PyCharm用）。

出力（out_quality/）:
    distribution.png        指標ごとのヒストグラム（Human／Robot × 質問者／回答者を、線で重ねたもの）
    distribution_by_group.png  同じ指標を、グループごとに別のグラフにしたもの（重ならず見やすい）
    distribution_stats.csv  同じ内容を数値にしたもの（パーセンタイルなど）
    reason_breakdown.csv    「要確認の理由」ごとの発話数と割合

準備: pip install matplotlib（PyCharm の Terminal タブで実行）
使い方: PyCharm で右クリック → Run
"""
import csv
import statistics
import sys
from pathlib import Path

# ============== 【設定】 ==============
MIN_UTTERANCES = 6           # これ以下の発話数のファイルは除く（話者が1人だけのファイルなど）
EXCLUDE = ["DAR25_007_Human", "DAR25_008_Human", "DAR25_031_Human", "DAR25_047_Human"]
# =====================================

BASE = Path(__file__).resolve().parent
QUALITY_DIR = BASE / "out_quality"

GROUPS = ["Human・回答者", "Human・質問者", "Robot・回答者", "Robot・質問者"]
COLORS = {"Human・回答者": "#d95f02", "Human・質問者": "#e6ab02", "Robot・回答者": "#1b9e77", "Robot・質問者": "#7570b3"}
REASONS = ["食い違い大", "低確信", "低確信(単語)", "Whisper警告", "重なり", "文が不自然"]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load() -> dict[str, list[dict]]:
    data: dict[str, list[dict]] = {g: [] for g in GROUPS}
    for p in sorted(QUALITY_DIR.glob("*_quality.csv")):
        base = p.name[: -len("_quality.csv")]
        if base in EXCLUDE:
            continue
        rows = read_csv(p)
        if len(rows) < MIN_UTTERANCES:
            continue
        kind = "Human" if base.endswith("Human") else "Robot" if base.endswith("Robot") else None
        if kind is None:
            continue
        for r in rows:
            g = f"{kind}・{r.get('役割', '')}"
            if g in data:
                data[g].append(r)
    return data


def pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def main() -> int:
    if not QUALITY_DIR.exists():
        print(f"{QUALITY_DIR} がありません（先に check_quality.py を実行）", file=sys.stderr)
        return 1
    data = load()
    if not any(data.values()):
        print("対象の発話がありません", file=sys.stderr)
        return 1

    metrics = [("食い違い度", "食い違い度", 0.0, 1.0, 20), ("単語確信度(平均)", "単語確信度(平均)", 0.0, 1.0, 20),
               ("単語確信度(最小)", "単語確信度(最小)", 0.0, 1.0, 20), ("重なり率（0は除く）", "重なり率", 0.0, 1.0, 20)]
    stats_rows = []
    values: dict[tuple[str, str], list[float]] = {}
    for g in GROUPS:
        for title, col, *_ in metrics:
            v = [x for x in (num(r.get(col)) for r in data[g]) if x is not None]
            if col == "重なり率":
                v = [x for x in v if x > 0]
            values[(g, title)] = v
            if v:
                stats_rows.append({"グループ": g, "指標": title, "件数": len(v), "平均": f"{statistics.mean(v):.2f}",
                                   "中央値": f"{statistics.median(v):.2f}", "25%": f"{pct(v, .25):.2f}",
                                   "75%": f"{pct(v, .75):.2f}", "90%": f"{pct(v, .90):.2f}"})
    with (QUALITY_DIR / "distribution_stats.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["グループ", "指標", "件数", "平均", "中央値", "25%", "75%", "90%"])
        w.writeheader()
        w.writerows(stats_rows)

    # 理由の内訳
    reason_rows = []
    for g in GROUPS:
        n = len(data[g])
        for reason in REASONS + ["（理由なし）"]:
            if reason == "（理由なし）":
                c = sum(1 for r in data[g] if not r.get("要確認の理由"))
            else:
                c = sum(1 for r in data[g] if reason in r.get("要確認の理由", "").split())
            reason_rows.append({"グループ": g, "理由": reason, "発話数": c, "割合": f"{c / n:.1%}" if n else ""})
    with (QUALITY_DIR / "reason_breakdown.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["グループ", "理由", "発話数", "割合"])
        w.writeheader()
        w.writerows(reason_rows)

    # 図
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAexGothic", "Hiragino Sans"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, (title, col, lo, hi, bins) in zip(axes.flat[:4], metrics):
        for g in GROUPS:
            v = values[(g, title)]
            if v:  # 塗りつぶさず、線だけで重ねる（重なっても見分けやすい）
                ax.hist(v, bins=bins, range=(lo, hi), histtype="step", linewidth=2.2, color=COLORS[g],
                        density=True, label=f"{g}（n={len(v)}）")
        ax.set_title(title)
        ax.set_xlabel("値")
        ax.set_ylabel("密度")
        ax.legend(fontsize=8)

    ax = axes.flat[4]  # 要確認の数
    maxk = 6
    width = 0.2
    for i, g in enumerate(GROUPS):
        n = len(data[g]) or 1
        counts = [sum(1 for r in data[g] if min(int(r.get("要確認の数", "0") or 0), maxk) == k) / n for k in range(maxk + 1)]
        ax.bar([k + (i - 1.5) * width for k in range(maxk + 1)], counts, width=width, color=COLORS[g], label=g)
    ax.set_title("要確認の理由の数（0=問題なし）")
    ax.set_xlabel("理由の数")
    ax.set_ylabel("割合")
    ax.legend(fontsize=8)

    ax = axes.flat[5]  # 理由別の割合
    labels = REASONS
    for i, g in enumerate(GROUPS):
        n = len(data[g]) or 1
        rates = [sum(1 for r in data[g] if reason in r.get("要確認の理由", "").split()) / n for reason in labels]
        ax.bar([k + (i - 1.5) * width for k in range(len(labels))], rates, width=width, color=COLORS[g], label=g)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("理由ごとの割合（重複あり）")
    ax.set_ylabel("割合")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(QUALITY_DIR / "distribution.png", dpi=130)

    # グループごとに分けた図（行=グループ、列=指標）。色が重ならないので、形を比べやすい
    fig2, axs = plt.subplots(len(GROUPS), len(metrics), figsize=(18, 11), sharex="col")
    for i, g in enumerate(GROUPS):
        for j, (title, col, lo, hi, bins) in enumerate(metrics):
            ax2 = axs[i][j]
            v = values[(g, title)]
            if v:
                ax2.hist(v, bins=bins, range=(lo, hi), color=COLORS[g], alpha=0.85)
                ax2.axvline(statistics.median(v), color="black", linestyle="--", linewidth=1)
                ax2.text(0.97, 0.92, f"中央値 {statistics.median(v):.2f}\nn={len(v)}", transform=ax2.transAxes,
                         ha="right", va="top", fontsize=9)
            if i == 0:
                ax2.set_title(title)
            if j == 0:
                ax2.set_ylabel(g)
            if i == len(GROUPS) - 1:
                ax2.set_xlabel("値")
    fig2.tight_layout()
    fig2.savefig(QUALITY_DIR / "distribution_by_group.png", dpi=130)
    print(f"出力先: {QUALITY_DIR}\n  distribution.png / distribution_by_group.png / distribution_stats.csv / reason_breakdown.csv")
    for g in GROUPS:
        n = len(data[g])
        if n:
            print(f"  {g}: {n}発話 / 理由なし {sum(1 for r in data[g] if not r.get('要確認の理由')) / n:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())