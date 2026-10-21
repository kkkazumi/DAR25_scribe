#!/usr/bin/env python3
"""
ロボット（ぼのちゃん）の実行ログ(.txt)から、ロボットが話した文章と、聞いた質問の番号を取り出す（PyCharm用）。

前提: robot_log/ フォルダに、面談ごとのログ（.txt）を置く。
      ファイル名に面談の番号（DAR25_001 など）が入っていればOK（例: DAR25_001.txt, DAR25_001__1_.txt）。
      questions.txt（44問）が、このファイルと同じ階層にある。

取り出すもの:
    ・「[Bono speaks] ...」の行 … ロボットが実際に話した文章（順番どおり）
    ・「入力を受け付けました: 名前 [36, 14, ...]」 … その面談で聞く質問の番号（0始まりなので、+1して使う）
    ・本番の前の動作確認（音量テスト・音声認識テスト）の発話は「テスト」として区別する
    ※ ロボット側のその場の文字起こし（[User(raw)] など）は、精度が低く幻覚も多いので使わない
    ※ ユーザ名（入力された名前）は、結果に残さず「○○」にする

出力（robot_log_parsed/）:
    <mp3名>_robot.csv     ロボットの発話の一覧（順番, 区分, ロボット発話, 質問ID, 質問(台本), 名前の疑い）
    robot_log_summary.csv 面談ごとの集計（その面談で聞く質問の番号、発話数 など）

使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import re
import sys
from pathlib import Path

from align_questions import bigrams, load_questions, normalize
from assign_roles import MASK_TEXT, mask_names

# ============== 【設定】 ==============
LOG_DIR_NAME = "robot_log"     # ログを置くフォルダ名
SUFFIX = "_Robot"              # 面談番号に付けて、mp3名にする（DAR25_001 → DAR25_001_Robot）
QUESTION_MATCH = 0.75          # 台本の質問の文字（2文字ずつ）が、この割合以上、発話に含まれていれば「その質問を聞いた」とする
# =====================================

BASE = Path(__file__).resolve().parent
LOG_DIR = BASE / LOG_DIR_NAME
OUT_DIR = BASE / "robot_log_parsed"

SPEAK = re.compile(r"^\[Bono speaks\]\s*(.*)$")
HEADER = re.compile(r"入力を受け付けました:\s*(.*?)\s*\[([\d,\s]+)\]")
TEST_DONE = "音声認識テスト完了"


def read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "cp932"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def question_of(text: str, qnorm_grams: list[set[str]], candidates: list[int]) -> int | None:
    """発話に含まれる台本の質問（候補の中で、含まれる割合が最も高く、しきい値以上のもの）"""
    ug = bigrams(normalize(text))
    best, best_id = 0.0, None
    for qid in candidates:
        g = qnorm_grams[qid - 1]
        if not g:
            continue
        r = len(g & ug) / len(g)
        if r > best:
            best, best_id = r, qid
    return best_id if best >= QUESTION_MATCH else None


def last_sentence(text: str) -> str:
    parts = [p for p in re.split(r"(?<=[。！？!?])", text) if p.strip()]
    return parts[-1].strip() if parts else text


NAME_LINE = re.compile(r"^(.{1,8}?)(?:さん、|さんという|ちゃん、|くん、|という名前|という素敵)")


def learn_names(texts: list[str], user_name: str) -> set[str]:
    """名前を聞いた直後のロボットの発話（「○○さん、素敵なお名前ですね」）から、呼んでいる名前を拾う"""
    names = {user_name} if len(user_name) >= 2 else set()
    for i, t in enumerate(texts[:-1]):
        if "お名前" in t:
            m = NAME_LINE.match(texts[i + 1].strip())
            if m and m.group(1) != MASK_TEXT and len(m.group(1)) >= 2:
                names.add(m.group(1))
    return names


def parse_one(path: Path, questions: list[str], qgrams: list[set[str]]) -> tuple[list[dict], dict]:
    """戻り値の info["_names"] に、ログから拾った名前（伏せ字にする対象）が入る"""
    lines = read_text(path).splitlines()
    chosen: list[int] = []
    user_name = ""
    for ln in lines:
        m = HEADER.search(ln)
        if m:
            user_name = m.group(1).strip()
            chosen = [int(x) + 1 for x in m.group(2).replace(" ", "").split(",") if x != ""]  # 0始まり → 1始まり
            break
    has_marker = any(TEST_DONE in ln for ln in lines)
    started = not has_marker  # テスト終了の印がなければ、全部を本番とみなす

    raw_texts = [m.group(1).strip() for m in (SPEAK.match(ln) for ln in lines) if m]
    names = learn_names(raw_texts, user_name)
    out: list[dict] = []
    for ln in lines:
        if TEST_DONE in ln:
            started = True
        m = SPEAK.match(ln)
        if not m:
            continue
        text = m.group(1).strip()
        for nm in sorted(names, key=len, reverse=True):
            text = text.replace(nm, MASK_TEXT)
        text = mask_names(text)[0]
        qid = question_of(text, qgrams, chosen) if chosen else None
        if qid is None:
            qid = question_of(text, qgrams, list(range(1, len(questions) + 1)))
        extra = ""
        if qid is None and re.search(r"[？?]\s*$", text):
            extra = "追加質問"
        out.append({
            "順番": len(out) + 1, "区分": "本番" if started else "テスト", "ロボット発話": text,
            "質問ID": qid if qid is not None else extra,
            "質問(台本)": questions[qid - 1] if qid is not None else (last_sentence(text) if extra else ""),
            "名前の疑い": "●" if re.search(r"お名前|という名前|さん", text) else "",
        })
    info = {"_names": sorted(names), "質問番号(ログ)": " ".join(map(str, chosen)), "本番の発話数": sum(1 for o in out if o["区分"] == "本番"),
            "テストの発話数": sum(1 for o in out if o["区分"] == "テスト"),
            "質問を含む発話数": sum(1 for o in out if isinstance(o["質問ID"], int))}
    return out, info


def main() -> int:
    if not LOG_DIR.exists():
        print(f"{LOG_DIR} がありません。ログ(.txt)を入れたフォルダを作ってください。", file=sys.stderr)
        return 1
    files = sorted(LOG_DIR.glob("*.txt"))
    if not files:
        print(f"{LOG_DIR} に .txt がありません", file=sys.stderr)
        return 1
    questions = load_questions()
    qgrams = [bigrams(normalize(q)) for q in questions]
    OUT_DIR.mkdir(exist_ok=True)
    summary, private = [], []
    for p in files:
        m = re.search(r"(DAR25_\d+)", p.name)
        if not m:
            print(f"[WARN] 面談の番号が分からないファイル名です: {p.name}")
            continue
        base = m.group(1) + SUFFIX
        rows, info = parse_one(p, questions, qgrams)
        if not rows:
            print(f"[WARN] {p.name}: [Bono speaks] の行がありません")
            continue
        with (OUT_DIR / f"{base}_robot.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        for nm in info["_names"]:
            private.append({"mp3名": base, "名前": nm})
        summary.append({"mp3名": base, "ログのファイル": p.name, **{k: v for k, v in info.items() if k != "_names"}})
        print(f"[{base}] ロボット発話 {len(rows)}件（本番 {info['本番の発話数']}／テスト {info['テストの発話数']}）"
              f" / 質問番号(ログ): {info['質問番号(ログ)']} / 質問を含む発話 {info['質問を含む発話数']}件")
    if summary:
        with (OUT_DIR / "robot_log_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    if private:  # 音声・下書きで名前を伏せるための一覧。実名が入っているので、共有しない
        with (OUT_DIR / "robot_names_PRIVATE.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["mp3名", "名前"])
            w.writeheader()
            w.writerows(private)
    print(f"\n出力先: {OUT_DIR}\n（名前の疑い ● の行は、ロボットが参加者の名前を呼んでいないか、目で確認してください）")
    return 0


if __name__ == "__main__":
    sys.exit(main())