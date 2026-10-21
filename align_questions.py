#!/usr/bin/env python3
"""
質問項目（questions.txt）と、各発話を対応づける（PyCharm用）。

やること:
    1. 質問者の発話を、質問項目と照合して「質問ID」を付ける
       （Whisper の文字起こしの類似度で判定。Whisper がなければ Azure の文字）
    2. 十分一致した質問者の発話は、台本の文に置き換える。一致しなかった質問者の発話は、
       Whisper の文からフィラーを除いたものにする
    3. 回答者の発話に、直前の質問IDを付ける
    4. 名前（「○○さん」）を伏せ字にする

前提:
    split_audio.py が済んでいて、csv_merged/*_merged.csv がある
    questions.txt（1行1問。行番号=質問ID）をこのファイルと同じ階層に置く
    あれば使うもの: out_roles/respondent_decision.csv（回答者の判定）、
                    out_roles/<mp3名>_roles.csv、out_transcript/<mp3名>_transcript.csv（transcribe.py の出力）

出力（out_aligned/）:
    <mp3名>_aligned.csv      全発話に質問IDなどを付けたもの
    alignment_summary.csv    ファイルごとの集計（検出できなかった質問IDなど）
    question_review.csv      要確認の質問者の発話（候補つき）

使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import bisect
import csv
import difflib
import math
import re
import sys
import unicodedata
from pathlib import Path

from assign_roles import QUESTION_END, infer_respondent, mask_names, read_csv, spk_num
from transcribe import FILLER_RE

# ============== 【設定】 ==============
TARGETS = []            # 処理する mp3 名（拡張子なし）。[] なら csv_merged 内のすべて
ACCEPT = 0.55           # この類似度以上で「質問一致」とする
REVIEW = 0.35           # この類似度以上で「要確認（候補あり）」とする
MIN_MARGIN = 0.05       # 1位と2位の差がこれ未満なら「要確認」
ACCEPT_LOW = 0.40       # 1位と2位の差が十分大きいときは、この類似度以上でも「質問一致」とする
ACCEPT_MARGIN = 0.25    #   （漢字・仮名の違いで類似度が下がった場合の救済）
ORDER_BONUS = 0.10      # 前の質問の「次の番号」の質問に足す加点。質問順がランダムなら 0
REPLACE_WITH_SCRIPT = True   # 一致した質問者の発話を、台本の文に置き換える
RESPONDENT_TEXT = "hybrid"   # 回答者の「発言内容(修正後)」に使う文字起こし。
                             # "hybrid" : Whisper の文字に、Azure のフィラー（えっと、あの 等）を戻したもの
                             # "whisper": Whisper の結果（なければ Azure） / "azure": Azure の結果
TRANSFER_PUNCT = False       # True: Whisper の文に、Azure の句読点を（文字が一致する箇所だけ）移す
PUNCT_FROM_PAUSE = True      # True: Whisper の単語ごとの時刻（_words.csv）から、間の長さで句読点を入れる
COMMA_GAP = 0.30             #   この秒数以上の間があれば「、」
PERIOD_GAP = 0.70            #   この秒数以上の間があれば「。」（直前が「か」なら「？」）
# =====================================

BASE = Path(__file__).resolve().parent
QUESTIONS = BASE / "questions.txt"
MERGED_DIR = BASE / "csv_merged"
ROLES_DIR = BASE / "out_roles"
TRANS_DIR = BASE / "out_transcript"
OUT_DIR = BASE / "out_aligned"


# --- 文字の正規化と類似度 ---
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\s、。，．,.！!？?「」『』（）()・…～~\-]", "", s)
    # カタカナ → ひらがな（表記ゆれ対策）
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else set(s)


class QuestionIndex:
    def __init__(self, questions: list[str]):
        self.questions = questions
        self.norm = [normalize(q) for q in questions]
        self.grams = [bigrams(q) for q in self.norm]
        n = len(questions)
        df: dict[str, int] = {}
        for g in self.grams:
            for b in g:
                df[b] = df.get(b, 0) + 1
        # 多くの質問に共通する語（「ありますか」など）の重みを下げる
        self.idf = {b: math.log((n + 1) / c) for b, c in df.items()}
        self.total = [sum(self.idf[b] for b in g) or 1.0 for g in self.grams]

    def scores(self, text: str) -> list[float]:
        tg = bigrams(normalize(text))
        return [sum(self.idf[b] for b in g if b in tg) / t for g, t in zip(self.grams, self.total)]


FILLER_SPAN_RE = re.compile("(?:" + FILLER_RE.pattern + ")[、。,.]?")
_PUNCT = set("、。,.，．！？!? \t　")


def _strip_punct(s: str) -> tuple[str, list[int]]:
    """句読点・空白を除いた文字列と、各文字の元の位置"""
    kept = [(c, i) for i, c in enumerate(s) if c not in _PUNCT]
    return "".join(c for c, _ in kept), [i for _, i in kept]


_PUNCT_MARKS = set("、。，．,.？?！!")


def transfer_punct(whisper: str, azure: str) -> str:
    """Azure の句読点を、対応する位置の Whisper の文字の後ろに移す（Whisper にすでにある句読点はそのまま）。
    文字が一致した箇所の直後の句読点だけ移すので、一致しない箇所には何も足さない。"""
    azure = FILLER_SPAN_RE.sub("", azure)  # フィラーは別途戻すので、句読点の位置合わせでは除いておく
    a_np, a_idx = _strip_punct(azure)
    w_np, w_idx = _strip_punct(whisper)
    if not a_np or not w_np:
        return whisper
    # Azure 側: 句読点なしの文字 k の直後にある句読点（連続はまとめる）
    after: dict[int, str] = {}
    for k, pos in enumerate(a_idx):
        j = pos + 1
        marks = ""
        while j < len(azure) and azure[j] in _PUNCT_MARKS:
            marks += azure[j]
            j += 1
        if marks:
            after[k] = marks
    ops = difflib.SequenceMatcher(None, a_np, w_np, autojunk=False).get_opcodes()
    a2w: dict[int, int] = {}
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            for off in range(i2 - i1):
                a2w[i1 + off] = j1 + off
    add: dict[int, str] = {}  # Whisper の句読点なし位置 → 直後に足す句読点
    for k, marks in after.items():
        if k not in a2w:
            continue
        j = a2w[k]
        # 句読点の前後の文字がどちらも一致している箇所（または文末）にだけ足す
        at_end = k == len(a_np) - 1 and j == len(w_np) - 1
        both = (k + 1) in a2w and a2w[k + 1] == j + 1
        if at_end or both:
            add[j] = marks
    out = []
    for jn, pos in enumerate(w_idx):
        out.append(whisper[pos])
        nxt = pos + 1
        has_punct = nxt < len(whisper) and whisper[nxt] in _PUNCT_MARKS
        # 元の Whisper の句読点を、そのまま残す
        k = nxt
        while k < len(whisper) and whisper[k] in _PUNCT_MARKS:
            out.append(whisper[k])
            k += 1
        if jn in add and not has_punct:
            out.append(add[jn])
    # 先頭の句読点など、文字の前にあるものを保持
    lead = whisper[: w_idx[0]] if w_idx else ""
    res = lead + "".join(out)
    return re.sub(r"([、，,])[、，,]+", r"\1", res)


def load_words(path: Path) -> dict[str, list[tuple[str, float, float]]]:
    out: dict[str, list[tuple[str, float, float]]] = {}
    if path.exists():
        for r in read_csv(path):
            out.setdefault(r["ファイル名"], []).append((r["単語"], float(r["開始(相対秒)"]), float(r["終了(相対秒)"])))
    return out


def punctuate_from_words(words: list[tuple[str, float, float]], fallback: str) -> str:
    """Whisper の単語列から、単語間の間の長さで句読点を入れた文を作る。
    単語をつないだ文字が文字起こし本文と一致しない場合は、本文（fallback）をそのまま返す。"""
    if not words:
        return fallback
    joined = "".join(w for w, _, _ in words)
    if _strip_punct(joined)[0] != _strip_punct(fallback)[0]:
        return fallback
    out = []
    for i, (w, st, en) in enumerate(words):
        out.append(w)
        last = w.rstrip()[-1:] if w.strip() else ""
        if i + 1 < len(words) and last not in _PUNCT_MARKS:
            gap = words[i + 1][1] - en
            if gap >= PERIOD_GAP:
                out.append("？" if last == "か" else "。")
            elif gap >= COMMA_GAP:
                out.append("、")
    res = "".join(out)
    if res and res[-1] not in _PUNCT_MARKS:
        res += "？" if res[-1] == "か" else "。"
    return res


def restore_fillers(whisper: str, azure: str) -> tuple[str, list[str]]:
    """Whisper の文字に、Whisper が省いた Azure のフィラーを、元の位置に戻す。
    句読点を無視して位置を合わせ、戻したフィラーの一覧も返す。"""
    a_np, a_idx = _strip_punct(azure)
    w_np, w_idx = _strip_punct(whisper)
    ops = difflib.SequenceMatcher(None, a_np, w_np, autojunk=False).get_opcodes()

    def to_orig(j: int) -> int:  # 句読点なしの位置 → Whisper 側の元の位置
        return w_idx[j] if j < len(w_idx) else len(whisper)

    inserts: list[tuple[int, str]] = []
    for m in FILLER_SPAN_RE.finditer(azure):
        n0 = bisect.bisect_left(a_idx, m.start())
        n1 = bisect.bisect_left(a_idx, m.end())
        if n1 <= n0:
            continue
        matched, target = 0, None
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                matched += max(0, min(n1, i2) - max(n0, i1))
            elif target is None and i1 <= n0 < i2:
                target = (j1, j2)
        if matched >= 0.5 * (n1 - n0):
            continue  # Whisper 側に、すでに残っている
        if target is None:  # 途中から一致している場合などは、一致の手前の位置に戻す
            pos = next((j1 for tag, i1, i2, j1, j2 in ops if i1 <= n0 <= i2), len(w_np))
            target = (pos, pos)
        j1, j2 = target
        lo, hi = to_orig(max(0, j1 - 4)), to_orig(min(len(w_np), j2 + 4))
        if FILLER_RE.search(whisper[lo:hi + 1]):
            continue  # 近くに別の表記のフィラー（えーと⇔えっと）がある
        inserts.append((to_orig(j1), m.group(0)))
    out = whisper
    for pos, f in sorted(inserts, reverse=True):
        out = out[:pos] + f + out[pos:]
    return out, [f for _, f in sorted(inserts)]


def load_questions() -> list[str]:
    if not QUESTIONS.exists():
        raise FileNotFoundError(f"{QUESTIONS} がありません")
    for enc in ("utf-8-sig", "cp932"):
        try:
            lines = QUESTIONS.read_text(encoding=enc).splitlines()
            return [ln.strip() for ln in lines if ln.strip()]
        except UnicodeDecodeError:
            continue
    raise ValueError("questions.txt の文字コードを判定できません")


def rank(index: QuestionIndex, text: str, prev_q: int) -> list[tuple[float, float, int]]:
    """[(順番の加点を含むスコア, 生の類似度, 質問ID)] を高い順に返す"""
    sc = index.scores(text)
    out = []
    for i, s in enumerate(sc, start=1):
        bonus = ORDER_BONUS if prev_q and i == prev_q + 1 else 0.0
        out.append((s + bonus, s, i))
    return sorted(out, reverse=True)


def respondent_of(base: str, rows: list[dict]) -> tuple[int, str]:
    p = ROLES_DIR / "respondent_decision.csv"
    if p.exists():
        for r in read_csv(p):
            if r["mp3名"] == base:
                return spk_num(r["回答者"]), "respondent_decision.csv"
    resp, method, _ = infer_respondent(rows)
    return resp, f"推定（{method}）"


def load_map(path: Path, key: str, value: str) -> dict[str, str]:
    return {r[key]: r[value] for r in read_csv(path)} if path.exists() else {}


def main() -> int:
    questions = load_questions()
    index = QuestionIndex(questions)
    print(f"質問項目: {len(questions)}問")

    csvs = sorted(MERGED_DIR.glob("*_merged.csv"))
    if not csvs:
        print(f"{MERGED_DIR} にマージ済みCSVがありません（先に split_audio.py を実行）", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(exist_ok=True)
    summary, review_rows = [], []

    for csv_path in csvs:
        rows = read_csv(csv_path)
        if not rows:
            continue
        base = re.sub(r"_\d+_spk\d+\.mp3$", "", rows[0]["ファイル名"])
        if TARGETS and base not in TARGETS:
            continue
        resp, how = respondent_of(base, rows)

        masked_resp = load_map(ROLES_DIR / f"{base}_roles.csv", "ファイル名", "発話内容")
        whisper = load_map(TRANS_DIR / f"{base}_transcript.csv", "ファイル名", "文字起こし(Whisper)")
        if PUNCT_FROM_PAUSE:
            wmap = load_words(TRANS_DIR / f"{base}_words.csv")
            whisper = {k: punctuate_from_words(wmap.get(k, []), v) if v else v for k, v in whisper.items()}

        prev_q, cur_q, pending, after_extra = 0, None, False, False
        seen: dict[int, int] = {}
        in_q_count: dict[int, int] = {}
        out = []
        for r in rows:
            spk = spk_num(r["話者"])
            is_resp = spk == resp
            raw = r["発話内容"]
            rec = {"通し番号": r["通し番号"], "話者": r["話者"], "役割": "回答者" if is_resp else "質問者",
                   "開始": r["開始"], "終了": r["終了"], "ファイル名": r["ファイル名"],
                   "発言内容(元)": mask_names(raw)[0], "発言内容(修正後)": "", "修正後の出所": "", "補ったフィラー": "",
                   "質問ID": "", "質問(台本)": "",
                   "類似度": "", "判定": "", "要確認": "", "質問内の回答番号": "", "文字起こし(Whisper)": ""}
            if is_resp:
                azure_txt = masked_resp.get(r["ファイル名"]) or mask_names(raw)[0]
                whisper_txt = mask_names(whisper[r["ファイル名"]])[0] if whisper.get(r["ファイル名"]) else ""
                rec["文字起こし(Whisper)"] = whisper_txt  # 名前は伏せ字にする
                if RESPONDENT_TEXT == "hybrid" and whisper_txt:
                    punct_txt = transfer_punct(whisper_txt, azure_txt) if TRANSFER_PUNCT else whisper_txt
                    merged_txt, added = restore_fillers(punct_txt, azure_txt)
                    rec["発言内容(修正後)"] = merged_txt
                    rec["修正後の出所"] = "Whisper+Azureのフィラー" if added else "Whisper"
                    rec["補ったフィラー"] = " ".join(added)
                elif RESPONDENT_TEXT in ("hybrid", "whisper") and whisper_txt:
                    rec["発言内容(修正後)"], rec["修正後の出所"] = whisper_txt, "Whisper"
                else:
                    rec["発言内容(修正後)"], rec["修正後の出所"] = azure_txt, "Azure"
                rec["判定"] = "回答"
                if cur_q:
                    rec["質問ID"], rec["質問(台本)"] = cur_q, questions[cur_q - 1]
                    in_q_count[cur_q] = in_q_count.get(cur_q, 0) + 1
                    rec["質問内の回答番号"] = in_q_count[cur_q]
                if pending or not cur_q:
                    rec["要確認"] = "質問IDが不確か"
                elif after_extra:
                    rec["要確認"] = "質問項目以外の質問のあと（質問IDは直前の質問項目）"
            else:
                w_i = whisper.get(r["ファイル名"], "")
                w_i = mask_names(w_i)[0] if w_i else ""
                rec["文字起こし(Whisper)"] = w_i
                match_txt = w_i or raw
                clean_i = FILLER_SPAN_RE.sub("", w_i).strip() if w_i else ""
                alt_txt, alt_src = (clean_i, "Whisper(フィラー除去)") if clean_i else (rec["発言内容(元)"], "Azure")
                ranked = rank(index, match_txt, prev_q)
                (s1, raw1, q1), (s2, _, _) = ranked[0], ranked[1]
                rec["類似度"] = f"{raw1:.2f}"  # 順番の加点を含まない、生の類似度
                margin = s1 - s2
                if (s1 >= ACCEPT and margin >= MIN_MARGIN) or (s1 >= ACCEPT_LOW and margin >= ACCEPT_MARGIN):
                    cur_q, prev_q, pending, after_extra = q1, q1, False, False
                    seen[q1] = seen.get(q1, 0) + 1
                    rec.update({"質問ID": q1, "質問(台本)": questions[q1 - 1], "判定": "質問一致"})
                    close = len(normalize(match_txt)) <= 1.5 * len(index.norm[q1 - 1]) + 6
                    if REPLACE_WITH_SCRIPT and close:
                        rec["発言内容(修正後)"], rec["修正後の出所"] = questions[q1 - 1], "台本"
                        rec["判定"] = "質問一致（台本に置換）"
                    else:
                        rec["発言内容(修正後)"], rec["修正後の出所"] = alt_txt, alt_src
                elif s1 >= REVIEW:
                    pending = True
                    rec.update({"質問ID": f"候補{q1}", "質問(台本)": questions[q1 - 1], "判定": "要確認（候補あり）",
                                "要確認": "質問者の発話", "発言内容(修正後)": alt_txt, "修正後の出所": alt_src})
                    review_rows.append({"mp3名": base, "通し番号": r["通し番号"], "発言内容": alt_txt,
                                        "候補の質問ID": q1, "候補の質問": questions[q1 - 1], "類似度": f"{raw1:.2f}"})
                else:
                    rec.update({"判定": "追加発話（質問項目以外）", "発言内容(修正後)": alt_txt,
                                "修正後の出所": alt_src})
                    if QUESTION_END.search(match_txt) and len(normalize(match_txt)) >= 8:
                        after_extra = True  # 質問らしい追加の発話。言い換えた質問項目の可能性もある
            out.append(rec)

        cols = list(out[0].keys())
        with (OUT_DIR / f"{base}_aligned.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(out)

        missing = [i for i in range(1, len(questions) + 1) if i not in seen]
        dup = [i for i, c in seen.items() if c > 1]
        n_review = sum(1 for o in out if o["判定"].startswith("要確認"))
        summary.append({"mp3名": base, "回答者": f"話者{resp}（{how}）", "検出した質問数": len(seen),
                        "要確認の質問者発話": n_review,
                        "重複して検出した質問ID": " ".join(map(str, sorted(dup))),
                        "検出できなかった質問ID": " ".join(map(str, missing))})
        print(f"[{base}] 回答者=話者{resp} / 質問検出 {len(seen)}問 / 要確認 {n_review}件 / 重複 {len(dup)}件")

    for name, data in (("alignment_summary.csv", summary), ("question_review.csv", review_rows)):
        if data:
            with (OUT_DIR / name).open("w", encoding="utf-8-sig", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    print(f"\n出力先: {OUT_DIR}\n（alignment_summary.csv と question_review.csv を確認し、ACCEPT / REVIEW を調整してください）")
    return 0


if __name__ == "__main__":
    sys.exit(main())