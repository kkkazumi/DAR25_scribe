#!/usr/bin/env python3
"""
外注に渡す音声ファイルと、校正用の下書きテキスト、依頼先ごとのリストを作る（PyCharm用）。名前の伏せ字もここで入れる。

前提: triage_outsourcing.py、align_questions.py（Robotは parse_robot_log.py と align_robot_log.py も）まで済んでいる。
      ffmpeg が使えること（これまでの split_audio.py と同じ）。

【渡す音声】 面談ごとに1ファイル（ファイル数は少ないほうが依頼しやすく、タイムスタンプも不要なので）
    ・Human条件   … 元のmp3まるごと（質問者・回答者の両方）
    ・Robot条件   … （コエラボに出す要注意の面談も同じ）回答者の発言だけを、順番に連結した1ファイル（発言と発言のあいだに GAP_SEC 秒の無音）。
                    ロボットの発話はログで分かっているので、外注の対象に入れない（費用も作業も減る）
    ・名前の部分  … 音声の該当区間を無音にする（Whisperの単語の時刻から探す）。下書きテキストでは「○○」にする
【下書きテキスト】 データグリーンに校正してもらう文字起こし（タイムスタンプなし）
    ・Human … 「【質問者】…／【回答者】…」の行
    ・Robot … 回答者の発言だけ。1行 = 1つの発言（連結音声と同じ順番）
【コエラボ（素起こし）】 音声だけを渡す（下書きは渡さない）

名前を探す方法（完全ではないので、リストの「伏せた箇所」と下書きを目で確認してください）:
    1. 「○○さん」「○○くん」「○○ちゃん」など
    2. 「私は○○です」「○○と申します」「名前は○○」など、名乗りの言い方
    3. 「お名前」を聞かれた直後の、短い返事
    4. known_names.csv（mp3名, 名前 の2列。任意）に書いた名前、およびロボットのログから拾った名前
       （robot_log_parsed/robot_names_PRIVATE.csv。実名が入っているので、人に渡さないこと）

出力（out_handover/）:
    audio/<名前>.mp3 または audio/<名前>_回答者のみ.mp3 … 渡す音声（名前は無音）
    audio/<名前>_回答者のみ_index.csv                    … 連結音声のどこが、どの発言か（Robotのみ）
    draft_txt/*.txt                                      … データグリーン用の下書き
    データグリーン_校正_リスト.csv / コエラボ_素起こし_リスト.csv
使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from assign_roles import MASK_TEXT, NAME_RE, NOT_NAMES

# ============== 【設定】 ==============
GAP_SEC = 1.0              # Robotの連結音声で、発言のあいだに入れる無音の長さ（秒）
MARGIN_SEC = 0.10          # 名前の単語の前後に足す無音の長さ（秒）
MAKE_AUDIO = True          # False なら、リストと下書きだけ作る
SKIP_EXISTING_AUDIO = True # 作成済みの音声は作り直さない
HUMAN_DG_RESPONDENT_ONLY = True    # True: データグリーンに出すHumanも、Robotと同じく回答者の発言だけを連結して渡す
AFTER_NAME_MAX_CHARS = 14  # 「お名前」を聞かれた直後の返事が、この文字数以下なら名前とみなす
# =====================================

BASE = Path(__file__).resolve().parent
TRIAGE = BASE / "out_triage"
ALIGNED = BASE / "out_aligned"
TRANS = BASE / "out_transcript"
SPLIT = BASE / "out_mp3"
MP3 = BASE / "mp3"
OUT = BASE / "out_handover"
KNOWN = BASE / "known_names.csv"
ROBOT_NAMES = BASE / "robot_log_parsed" / "robot_names_PRIVATE.csv"

KANA = "ぁ-んァ-ヶー"
NAME_CH = rf"[一-龥々{KANA}]"
SELF_INTRO = re.compile(rf"(?:私は|わたしは|僕は|ぼくは|名前は|名前が|名字は)({NAME_CH}{{2,8}}?)(?=です|といいます|と言います|と申します|でございます|って言います|ですね|だよ)")
SAY_NAME = re.compile(rf"({NAME_CH}{{2,8}}?)(?=と申します|といいます|と言います|って言います)")
BARE_ANSWER = re.compile(rf"^(?:はい、?|ええ、?|あ、?|えっと、?)?(?:私は)?({NAME_CH}{{2,8}}?)(?:です|といいます|と申します)?[。、]?$")
NOT_SELF = {"元気", "大丈夫", "そう", "はい", "ええ", "お願い", "よろしく"}


def read(p: Path) -> list[dict]:
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write(p: Path, rows: list[dict]) -> None:
    with p.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def load_known() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for p in (KNOWN, ROBOT_NAMES):
        if p.exists():
            for r in read(p):
                nm = (r.get("名前") or "").strip()
                if len(nm) >= 2:
                    out.setdefault(r["mp3名"], set()).add(nm)
    return out


def name_spans(text: str, known: set[str], after_name_q: bool) -> list[tuple[int, int]]:
    """text の中の名前の位置（開始, 終了）の一覧"""
    spans: list[tuple[int, int]] = []
    for m in NAME_RE.finditer(text):
        if m.group(1) not in NOT_NAMES:
            spans.append((m.start(1), m.end(1)))
    for rx in (SELF_INTRO, SAY_NAME):
        for m in rx.finditer(text):
            if m.group(1) not in NOT_SELF:
                spans.append((m.start(1), m.end(1)))
    for nm in known:
        for m in re.finditer(re.escape(nm), text):
            spans.append((m.start(), m.end()))
    if after_name_q and len(text) <= AFTER_NAME_MAX_CHARS:
        m = BARE_ANSWER.match(text.strip())
        if m and m.group(1) not in NOT_SELF and not spans:
            off = text.find(m.group(1))
            spans.append((off, off + len(m.group(1))))
    spans.sort()
    merged: list[tuple[int, int]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def mask_text(text: str, spans: list[tuple[int, int]]) -> str:
    for a, b in reversed(spans):
        text = text[:a] + MASK_TEXT + text[b:]
    return text


def load_words(base: str) -> dict[str, list[dict]]:
    p = TRANS / f"{base}_words.csv"
    out: dict[str, list[dict]] = {}
    if p.exists():
        for r in read(p):
            out.setdefault(r["ファイル名"], []).append(r)
    return out


def mute_times(words: list[dict], spans: list[tuple[int, int]], key_s: str, key_e: str) -> list[tuple[float, float]]:
    """名前の位置に当たる単語の時刻（前後に余白をつける）"""
    pos, ranges = 0, []
    for w in words:
        t = w["単語"].strip()
        ranges.append((pos, pos + len(t)))
        pos += len(t)
    out = []
    for a, b in spans:
        hit = [i for i, (x, y) in enumerate(ranges) if x < b and y > a]
        if hit:
            s = float(words[hit[0]][key_s]) - MARGIN_SEC
            e = float(words[hit[-1]][key_e]) + MARGIN_SEC
            out.append((max(0.0, s), e))
    return out


def mute_filter(times: list[tuple[float, float]]) -> str:
    if not times:
        return ""
    expr = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in times)
    return f"volume=enable='{expr}':volume=0"


def ffmpeg(args: list[str]) -> None:
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])


def duration(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def after_name_flags(rows: list[dict]) -> list[bool]:
    flags, prev_q = [], False
    for r in rows:
        flags.append(prev_q and r["役割"] == "回答者")
        t = r.get("発言内容(修正後)", "") + r.get("発言内容(元)", "")
        if r["役割"] == "質問者":
            prev_q = "お名前" in t or "名前を" in t
        elif r["役割"] == "回答者":
            prev_q = False
    return flags


def prepare(base: str, rows: list[dict], known: set[str]) -> list[dict]:
    """各発話に、名前の位置・伏せた後の下書き・無音にする時刻を付ける"""
    words = load_words(base)
    flags = after_name_flags(rows)
    for r, fl in zip(rows, flags):
        text = r.get("発言内容(修正後)", "")
        sp = name_spans(text, known, fl)
        r["_draft"] = mask_text(text, sp)
        r["_nmask"] = len(sp)
        ws = words.get(r["ファイル名"], [])
        # 下書きでは見つからなくても、単語の並びから名前を探す（Whisperの文字起こしが元）
        wtext = "".join(w["単語"].strip() for w in ws)
        sp_w = name_spans(wtext, known, fl) if ws else []
        r["_mute_rel"] = mute_times(ws, sp_w, "開始(相対秒)", "終了(相対秒)") if sp_w else []
        r["_mute_abs"] = mute_times(ws, sp_w, "開始(絶対秒)", "終了(絶対秒)") if sp_w else []
        r["_nmask_audio"] = len(sp_w)
    return rows


def build_human_audio(base: str, rows: list[dict], dst: Path) -> int:
    src = MP3 / f"{base}.mp3"
    if not src.exists():
        raise FileNotFoundError(f"{src} がありません")
    times = sorted(t for r in rows for t in r["_mute_abs"])
    args = ["-i", str(src)]
    flt = mute_filter(times)
    if flt:
        args += ["-af", flt]
    ffmpeg([*args, "-c:a", "libmp3lame", "-q:a", "3", str(dst)])
    return len(times)


def build_robot_audio(base: str, rows: list[dict], dst: Path, index_path: Path) -> int:
    resp = [r for r in rows if r["役割"] == "回答者"]
    n_mute = 0
    idx = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        gap = tmp / "gap.wav"
        ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", f"{GAP_SEC}", str(gap)])
        listing, t_now = [], 0.0
        for k, r in enumerate(resp, start=1):
            piece = SPLIT / base / r["ファイル名"]
            if not piece.exists():
                continue
            w = tmp / f"p{k:03d}.wav"
            args = ["-i", str(piece)]
            flt = mute_filter(r["_mute_rel"])
            args += ["-af", (flt + "," if flt else "") + "aresample=44100", "-ac", "1", str(w)]
            ffmpeg(args)
            d = duration(w)
            listing.append(f"file '{w.as_posix()}'")
            listing.append(f"file '{gap.as_posix()}'")
            idx.append({"通し番号": r["通し番号"], "質問ID": r.get("質問ID", ""), "連結音声での開始(秒)": f"{t_now:.1f}",
                        "終了(秒)": f"{t_now + d:.1f}", "下書き": r["_draft"], "伏せた箇所": r["_nmask_audio"]})
            t_now += d + GAP_SEC
            n_mute += len(r["_mute_rel"])
        if not listing:
            raise FileNotFoundError(f"out_mp3/{base}/ に回答者の分割音声が見つかりません")
        lst = tmp / "list.txt"
        lst.write_text("\n".join(listing), encoding="utf-8")
        ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst), "-ac", "1", "-c:a", "libmp3lame", "-q:a", "4", str(dst)])
    if idx:
        write(index_path, idx)
    return n_mute


def pack_send_folders(dg: list[dict], kl: list[dict]) -> None:
    """依頼先ごとに、実際に送るファイルだけを1つのフォルダにまとめる（内部用の列や名前の一覧は入れない）"""
    for name in ("送付_データグリーン", "送付_コエラボ"):
        shutil.rmtree(OUT / name, ignore_errors=True)
    d_dir, k_dir = OUT / "送付_データグリーン", OUT / "送付_コエラボ"
    for sub in ("音声", "下書きテキスト"):
        (d_dir / sub).mkdir(parents=True, exist_ok=True)
    (k_dir / "音声").mkdir(parents=True, exist_ok=True)
    d_rows, k_rows = [], []
    for r in dg:
        a, t = OUT / "audio" / r["渡す音声"], OUT / "draft_txt" / r["下書きテキスト"]
        if not a.exists() or not t.exists():
            continue
        shutil.copy2(a, d_dir / "音声" / a.name)
        shutil.copy2(t, d_dir / "下書きテキスト" / t.name)
        d_rows.append({"音声ファイル": a.name, "下書きテキスト": t.name, "音声の長さ(秒)": r["音声の長さ(秒)"],
                       "行数": r["発話数"], "要注意": "○" if str(r["備考"]).startswith("【要注意】") else ""})
    for r in kl:
        a = OUT / "audio" / r["渡す音声"]
        if not a.exists():
            continue
        shutil.copy2(a, k_dir / "音声" / a.name)
        k_rows.append({"音声ファイル": a.name, "依頼内容": r["依頼内容"], "音声の長さ(秒)": r["音声の長さ(秒)"]})
    if d_rows:
        write(d_dir / "依頼リスト.csv", d_rows)
    if k_rows:
        write(k_dir / "依頼リスト.csv", k_rows)
    print(f"\n送付フォルダ: {d_dir.name}（音声 {len(d_rows)}件） / {k_dir.name}（音声 {len(k_rows)}件）")


def main() -> int:
    hp, rp = TRIAGE / "triage_Human.csv", TRIAGE / "triage_Robot.csv"
    if not hp.exists() or not rp.exists():
        print("先に triage_outsourcing.py を実行してください", file=sys.stderr)
        return 1
    if MAKE_AUDIO and not shutil.which("ffmpeg"):
        print("ffmpeg が見つかりません", file=sys.stderr)
        return 1
    (OUT / "draft_txt").mkdir(parents=True, exist_ok=True)
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    known_all = load_known()
    dg, kl = [], []
    jobs = [("Human", r) for r in read(hp)] + [("Robot", r) for r in read(rp)]
    nh_dg = sum(1 for k, r in jobs if k == "Human" and r["依頼先(案)"].startswith("データグリーン"))
    print(f"設定: HUMAN_DG_RESPONDENT_ONLY = {HUMAN_DG_RESPONDENT_ONLY}")
    print(f"triage の Human {sum(1 for k, _ in jobs if k == 'Human')}件のうち、データグリーン行き {nh_dg}件（これが『回答者のみ』音声になる）\n")
    for kind, tr in jobs:
        base, dest = tr["mp3名"], tr["依頼先(案)"]
        ap = ALIGNED / f"{base}_aligned.csv"
        is_dg = dest.startswith("データグリーン")
        single = "タイムスタンプ" in dest
        if not ap.exists() and not single:
            print(f"[SKIP] {base}: aligned.csv がありません")
            continue
        rows = prepare(base, read(ap), known_all.get(base, set())) if ap.exists() else []
        resp_only = kind == "Robot" or (is_dg and HUMAN_DG_RESPONDENT_ONLY)
        if not resp_only:
            audio_name = f"{base}.mp3"
        else:
            audio_name = f"{base}_回答者のみ.mp3"
        dst = OUT / "audio" / audio_name
        n_mute, err = 0, ""
        if MAKE_AUDIO:
            try:
                if SKIP_EXISTING_AUDIO and dst.exists() and dst.stat().st_size > 10000:
                    n_mute = -1
                elif not resp_only:
                    n_mute = build_human_audio(base, rows, dst)
                else:
                    n_mute = build_robot_audio(base, rows, dst, OUT / "audio" / f"{base}_回答者のみ_index.csv")
            except Exception as e:
                err = f"音声を作れませんでした: {e}"
                print(f"[ERROR] {base}: {err}")
        n_text = sum(r["_nmask"] for r in rows)
        try:
            dur = round(duration(dst)) if dst.exists() else ""
        except Exception:
            dur = ""
        if is_dg:
            if not resp_only:
                lines = [f"【{r['役割']}】{r['_draft']}" for r in rows if r["_draft"].strip()]
                n = len(rows)
            else:
                lines = [r["_draft"] for r in rows if r["役割"] == "回答者" and r["_draft"].strip()]
                n = len(lines)
            txt_name = f"{base}.txt" if not resp_only else f"{base}_回答者のみ.txt"
            (OUT / "draft_txt" / txt_name).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
            note = " ".join(x for x in (tr.get("理由", ""), tr.get("備考", "")) if x)
            if dest.endswith("（要注意）"):
                note = "【要注意】" + note
            dg.append({"区分": kind, "渡す音声": audio_name, "音声": ("作成済" if dst.exists() else "未作成") + (f" {err}" if err else ""),
                       "音声の長さ(秒)": dur, "下書きテキスト": txt_name, "発話数": n, "音声で無音にした名前": "" if n_mute < 0 else n_mute,
                       "下書きで伏せた名前": n_text, "備考": note})
        else:
            kl.append({"区分": kind, "依頼内容": dest.replace("コエラボ：", ""), "渡す音声": audio_name,
                       "音声": ("作成済" if dst.exists() else "未作成") + (f" {err}" if err else ""),
                       "音声の長さ(秒)": dur, "音声で無音にした名前": "" if n_mute < 0 else n_mute, "備考": tr.get("理由", "")})
        print(f"[{base}] {dest} / 音声 {dur}秒{'（前回の作成済みを再利用）' if n_mute < 0 else ''} / 名前を伏せた箇所: 音声 {n_mute if n_mute >= 0 else '-'} / 下書き {n_text}")
    if dg:
        write(OUT / "データグリーン_校正_リスト.csv", dg)
    if kl:
        write(OUT / "コエラボ_素起こし_リスト.csv", kl)
    pack_send_folders(dg, kl)
    print(f"\nデータグリーン（校正）: {len(dg)}件 / コエラボ（素起こし）: {len(kl)}件\n出力先: {OUT}")
    print("（名前の伏せ漏れがないか、下書きと、リストの『伏せた名前』を確認してください）")
    return 0


if __name__ == "__main__":
    sys.exit(main())