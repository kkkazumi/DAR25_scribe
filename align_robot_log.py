#!/usr/bin/env python3
"""
ロボットのログ（parse_robot_log.py の出力）を使って、Robotインタビューの「質問者(ロボット)」の発話を直す（PyCharm用）。

前提: align_questions.py と parse_robot_log.py まで済んでいる。
      out_aligned/DAR25_xxx_Robot_aligned.csv と robot_log_parsed/DAR25_xxx_Robot_robot.csv がある面談だけ処理する。

やること:
  1. CSVの質問者(ロボット)の各行を、ログの発話に「順番を守って」割り当てる（文字の重なりで判定）。
     ロボットの1つの発話が複数行に分かれていても、録音にない発話や、ログにない余分な行があっても対応する。
  2. 割り当てた行の「発言内容(修正後)」を、ログの文章に置き換える（出所＝ロボットログ）。
     1つの発話が複数行に分かれているときは、最初の行にログの全文を入れ、残りの行は空にする。
  3. ログの質問番号から「質問ID」「質問(台本)」を付ける。回答者の発話にも、直前の質問を引き継ぐ。
  4. 要確認を付ける:
       ・ログにあるのに録音で聞き取れていない発話（または一部だけ）
       ・ログに対応しないロボット側の行（途中で割り込んだ音、回答者の発話の取り違え など）
       ・名前を聞かれた直後の回答者の発話（名前が入っている。伏せ字を確認）

出力:
  out_aligned/<base>_aligned.csv      … 上書き（元のものは out_aligned_before_log/ に1回だけ退避）
  robot_log_check/<base>_log_match.csv … ログの発話ごとの対応（どの行に当たったか、録音で聞き取れた割合）
  robot_log_check/robot_log_match_summary.csv … 面談ごとの集計
何度実行しても同じ結果になる（文字起こし(Whisper)の列を元に毎回やり直す）。
使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import re
import shutil
import sys
from pathlib import Path

from align_questions import bigrams, normalize

# ============== 【設定】 ==============
TARGETS = []              # 処理する mp3 名。[] なら、ログのあるRobot面談すべて
MIN_COVER = 0.40          # 行の文字のうち、この割合以上がログの発話に含まれていれば、その発話の一部とみなす
MIN_GRAMS = 3             # 行の文字（2文字ずつ）がこれ未満の短い行（「はい」など）は、割り当てず、そのまま残す
LOG_COVER_OK = 0.60       # ログの発話のうち、録音で聞き取れた割合がこれ未満なら要確認
LOG_MATCH_MIN = 0.60      # 本番の発話のうち、録音で確認できた割合がこれ未満なら「そのログは別の面談のもの」とみなし、置き換えない
INCLUDE_TEST = True       # テスト発話（音量確認）も、録音に含まれていれば割り当てる
# =====================================

BASE = Path(__file__).resolve().parent
ALIGNED = BASE / "out_aligned"
BACKUP = BASE / "out_aligned_before_log"
LOGP = BASE / "robot_log_parsed"
CHECK = BASE / "robot_log_check"


def read_rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def assign(row_grams: list[set[str]], log_grams: list[set[str]]) -> list[int | None]:
    """順番を守って、行 → ログの発話 に割り当てる（長い行ほど重視する）。None は割り当てなし"""
    n, m = len(row_grams), len(log_grams)
    gain = [[0.0] * m for _ in range(n)]
    for i, g in enumerate(row_grams):
        if len(g) < MIN_GRAMS:
            continue
        for j, lg in enumerate(log_grams):
            c = len(g & lg) / len(g)
            if c >= MIN_COVER:
                gain[i][j] = len(g) * (c - 0.25)
    NEG = float("-inf")
    # 添字 k = 0 が「まだ割り当てなし」、k = j+1 が「最後に割り当てたログが j」
    best = [[NEG] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int | None]]] = [[(0, None)] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n):
        for k in range(m + 1):
            if best[i][k] == NEG:
                continue
            # 割り当てない
            if best[i][k] > best[i + 1][k]:
                best[i + 1][k] = best[i][k]
                back[i + 1][k] = (k, None)
            # ログ j（j+1 >= k つまり同じかそれ以降）に割り当てる
            for j in range(max(k - 1, 0), m):
                if gain[i][j] <= 0:
                    continue
                v = best[i][k] + gain[i][j]
                if v > best[i + 1][j + 1]:
                    best[i + 1][j + 1] = v
                    back[i + 1][j + 1] = (k, j)
    k = max(range(m + 1), key=lambda x: best[n][x])
    res: list[int | None] = [None] * n
    for i in range(n, 0, -1):
        pk, j = back[i][k]
        res[i - 1] = j
        k = pk
    return res


def process(base: str, rows: list[dict], logs: list[dict]) -> tuple[list[dict], list[dict], dict]:
    use = [l for l in logs if INCLUDE_TEST or l["区分"] == "本番"]
    log_text = [l["ロボット発話"] for l in use]
    log_grams = [bigrams(normalize(t)) for t in log_text]
    robot_idx = [i for i, r in enumerate(rows) if r["役割"] == "質問者"]

    def src(r: dict) -> str:
        return r.get("文字起こし(Whisper)") or r.get("発言内容(元)") or ""

    row_grams = [bigrams(normalize(src(rows[i]))) for i in robot_idx]
    res = assign(row_grams, log_grams)
    by_log: dict[int, list[int]] = {}
    for pos, j in enumerate(res):
        if j is not None:
            by_log.setdefault(j, []).append(robot_idx[pos])

    for r in rows:  # やり直しに備えて、前回の結果を消す
        r.setdefault("ログ発話の網羅率", "")
        r["ログ発話の網羅率"] = ""

    # 質問ID・要確認の付け直し
    robot_set = set(robot_idx)
    match_rows = []
    for j, l in enumerate(use):
        idxs = by_log.get(j, [])
        union: set[str] = set()
        for i in idxs:
            union |= bigrams(normalize(src(rows[i])))
        cover = len(union & log_grams[j]) / len(log_grams[j]) if log_grams[j] else 0.0
        match_rows.append({
            "mp3名": base, "ログの順番": l["順番"], "区分": l["区分"], "質問ID": l["質問ID"],
            "ロボット発話(ログ)": l["ロボット発話"],
            "対応した通し番号": " ".join(rows[i]["通し番号"] for i in idxs),
            "録音で聞き取れた割合": f"{cover:.2f}" if idxs else "0.00",
            "状態": (("テスト発話（録音なし）" if l["区分"] == "テスト" else "録音になし") if not idxs else ("一部のみ" if cover < LOG_COVER_OK else "OK")),
        })
        if not idxs:
            continue
        if cover < LOG_COVER_OK:
            # 一部しか一致しないときは、置き換えない（別の発話に置き換える事故を防ぐ）
            for i in idxs:
                rows[i]["ログ発話の網羅率"] = f"{cover:.2f}"
                rows[i]["要確認"] = "ログと一部しか一致しない（置換せず）"
            continue
        first = idxs[0]
        for n, i in enumerate(idxs):
            r = rows[i]
            r["ログ発話の網羅率"] = f"{cover:.2f}"
            if n == 0:
                r["発言内容(修正後)"], r["修正後の出所"] = l["ロボット発話"], "ロボットログ"
                r["判定"] = "ロボットログに置換" if len(idxs) == 1 else f"ロボットログに置換（{len(idxs)}行に分かれた発話）"
            else:
                r["発言内容(修正後)"], r["修正後の出所"] = "", "ロボットログ（前の行に含む）"
                r["判定"] = "ロボットログ（前の行に含む）"
            r["要確認"] = "" if cover >= LOG_COVER_OK else "ログの発話の一部しか録音で確認できない"
        q = l["質問ID"]
        if l["区分"] == "テスト":
            q = ""  # 音量確認などは質問として扱わない
        elif isinstance(q, str) and not q.isdigit() and q:
            q = "名前" if "お名前" in l["ロボット発話"] else ("趣味" if "趣味" in l["ロボット発話"] else q)
        if isinstance(q, str) and q.isdigit():
            rows[first]["質問ID"], rows[first]["質問(台本)"] = int(q), l["質問(台本)"]
            rows[first]["類似度"] = f"{cover:.2f}"
        elif q:  # 名前・趣味・その他の追加質問
            rows[first]["質問ID"], rows[first]["質問(台本)"] = q, l["質問(台本)"]
        for i in idxs[1:]:
            rows[i]["質問ID"], rows[i]["質問(台本)"] = rows[first]["質問ID"], rows[first]["質問(台本)"]
    # ログに対応しない（短くない）ロボット側の行
    assigned = {i for ids in by_log.values() for i in ids}
    n_extra = 0
    for pos, i in enumerate(robot_idx):
        if i not in assigned:
            r = rows[i]
            if len(row_grams[pos]) >= MIN_GRAMS:
                r["判定"], r["要確認"] = "ログに対応しないロボット側の発話", "ログに無い発話（取り違え・割り込みの疑い）"
                n_extra += 1
            else:
                r["判定"] = r["判定"] or "短い発話（ログと未対応）"
    # 回答者の行に、直前の質問を引き継ぐ
    cur_id, cur_text, after_name = "", "", False
    counts: dict = {}
    name_q = {l["順番"] for l in use if "お名前" in l["ロボット発話"]}
    by_row_log = {rows[i]["通し番号"]: j for j, ids in by_log.items() for i in ids[:1]}
    for i, r in enumerate(rows):
        if i in robot_set:
            if r["質問ID"] != "":
                cur_id, cur_text = r["質問ID"], r["質問(台本)"]
                counts[cur_id] = 0
            j = by_row_log.get(r["通し番号"])
            if j is not None:
                after_name = use[j]["順番"] in name_q
            continue
        r["質問ID"], r["質問(台本)"] = cur_id, cur_text
        if cur_id != "":
            counts[cur_id] = counts.get(cur_id, 0) + 1
            r["質問内の回答番号"] = counts[cur_id]
        r["要確認"] = ""
        if cur_id == "":
            r["要確認"] = "最初の質問の前の発話（挨拶・名前など）"
        if after_name:
            r["要確認"] = "名前を聞かれた直後の回答（名前が入っていないか確認）"
            after_name = False
    info = {"mp3名": base, "ログの発話数": len(use),
            "録音で確認できた": sum(1 for m in match_rows if m["状態"] == "OK"),
            "一部のみ": sum(1 for m in match_rows if m["状態"] == "一部のみ"),
            "録音になし": sum(1 for m in match_rows if m["状態"] == "録音になし"),
            "テスト発話（録音なし）": sum(1 for m in match_rows if m["状態"] == "テスト発話（録音なし）"),
            "ログに無いロボット側の行": n_extra}
    return rows, match_rows, info


def main() -> int:
    if not LOGP.exists():
        print(f"{LOGP} がありません（先に parse_robot_log.py を実行）", file=sys.stderr)
        return 1
    CHECK.mkdir(exist_ok=True)
    BACKUP.mkdir(exist_ok=True)
    summary = []
    for lp in sorted(LOGP.glob("*_robot.csv")):
        base = lp.name[:-len("_robot.csv")]
        if TARGETS and base not in TARGETS:
            continue
        ap = ALIGNED / f"{base}_aligned.csv"
        if not ap.exists():
            print(f"[SKIP] {base}: {ap.name} がありません")
            continue
        with ap.open(encoding="utf-8-sig", newline="") as f:
            already = "ログ発話の網羅率" in (f.readline() or "")
        if already is False or not (BACKUP / ap.name).exists():
            shutil.copy2(ap, BACKUP / ap.name)  # align_questions.py をやり直した直後は、新しいものを元にする
        rows = read_rows(BACKUP / ap.name)  # 毎回、退避した元のものからやり直す
        rows, match_rows, info = process(base, rows, read_rows(lp))
        real = [m for m in match_rows if m["区分"] == "本番"]
        ok_frac = sum(1 for m in real if m["状態"] == "OK") / len(real) if real else 0.0
        info["確認できた割合"] = f"{ok_frac:.2f}"
        info["ログ不一致の疑い"] = ""
        if ok_frac < LOG_MATCH_MIN:
            # 録音とログの内容が大きく違う → 別の面談のログか、途中で中断した可能性。置き換えず、元のままにする
            rows = read_rows(BACKUP / ap.name)
            for r in rows:
                r["ログ発話の網羅率"] = ""
            info["ログ不一致の疑い"] = "●（置き換えなし）"
        write_rows(ap, rows)
        write_rows(CHECK / f"{base}_log_match.csv", match_rows)
        summary.append(info)
        print(f"[{base}] ログ{info['ログの発話数']}件: 確認できた{info['録音で確認できた']} / 一部のみ{info['一部のみ']}"
              f" / 録音になし{info['録音になし']}（ほかにテスト{info['テスト発話（録音なし）']}） / ログに無い行{info['ログに無いロボット側の行']}"
              + ("  ★ログが録音と合わないため置き換えなし" if info["ログ不一致の疑い"] else ""))
    if summary:
        write_rows(CHECK / "robot_log_match_summary.csv", summary)
    print(f"\n出力先: {CHECK}\n次は check_quality.py を実行し直してください（out_aligned を読み直します）")
    return 0


if __name__ == "__main__":
    sys.exit(main())