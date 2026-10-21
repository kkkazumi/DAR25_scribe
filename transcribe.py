#!/usr/bin/env python3
"""
質問者・回答者すべての発話mp3を、ローカルのWhisperで文字起こしする（PyCharm用）。音声は外部に送信しません。

前提: split_audio.py と assign_roles.py が済んでいて、out_mp3/<mp3名>/*.mp3 と
      out_roles/<mp3名>_roles.csv ができている。

準備:
    pip install openai-whisper          （PyCharm の Terminal タブで実行）
    ffmpeg が PATH に通っていること
    ※ 初回だけ、モデルのファイルをダウンロードします（音声は送信されません）

出力（out_transcript/）:
    <mp3名>_transcript.csv  発話ごとの文字起こし（Azureの元の文字起こしと並べて比較できる）
    <mp3名>_words.csv       単語ごとの時刻（ファイル内の相対時刻と、元音声での絶対時刻）

使い方: 下の【設定】を確認して、PyCharm で右クリック → Run
"""
import csv
import re
import shutil
import sys
import time
from pathlib import Path

# ============== 【設定】 ==============
TARGETS = []              # 処理する mp3 名（拡張子なし）。[] なら out_roles 内のすべて
MODEL_NAME = "large-v3"   # "large-v3"（高精度）/ "medium"（軽い）など
LANGUAGE = "ja"
INITIAL_PROMPT = None     # 専門用語を当てたいときの短い文。例: "高齢者の生活についてのインタビュー。"
KEEP_FILLERS = True       # True: 回答者の発話に、「えっと」「あの」などのフィラーを残しやすくするヒントを渡す（質問者には渡さない）
FILLER_PROMPT = "えーっと、あのー、うーん、えー、そのー、まあ、なんかこう、えっと。"
SKIP_EXISTING = True      # 出力済みのファイルは飛ばす（途中から再開できる）
PAD_SEC = 0.0             # split_audio.py の PAD_SEC と同じ値にする
# =====================================

BASE = Path(__file__).resolve().parent
ROLES_DIR = BASE / "out_roles"
SPLIT_DIR = BASE / "out_mp3"
OUT_DIR = BASE / "out_transcript"

# 無音部分で Whisper が出しがちな定型句（要確認フラグを立てる）
HALLUCINATIONS = ("ご視聴ありがとうございました", "ご視聴ありがとうございます", "ご視聴いただき",
                  "チャンネル登録", "最後までご視聴", "字幕", "高評価", "おやすみなさい",
                  "ご清聴ありがとうございました", "次の動画", "ありがとうございました。ありがとうございました")
DROP_HALLUCINATIONS = True   # True: 上の定型句を含む区間や、無音で出た低確信の区間を取り除く（Azureの文字に切り替わる）


FILLER_RE = re.compile(
    r"えー*っ?と|ええと|あのー+|あの(?=[、。,.\s])|えー+|うーん+|んー+|そのー+|その(?=[、。,.])|まあ(?=[、。,.])"
)


def count_fillers(text: str) -> int:
    return len(FILLER_RE.findall(text))


def only_fillers(text: str) -> bool:
    """フィラーと句読点だけの出力（ヒントをそのまま出力した幻覚の疑い）"""
    rest = FILLER_RE.sub("", re.sub(r"[\s、。,.]", "", text))
    return bool(text.strip()) and rest == "" and count_fillers(text) >= 3


def effective_prompt(is_resp: bool = True) -> str | None:
    parts = [p for p in (INITIAL_PROMPT, FILLER_PROMPT if (KEEP_FILLERS and is_resp) else None) if p]
    return "".join(parts) or None


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


def load_model():
    import whisper

    try:
        import torch
        use_cuda = torch.cuda.is_available()
    except ImportError:
        use_cuda = False
    device = "cuda" if use_cuda else "cpu"
    print(f"Whisper {MODEL_NAME} を読み込み中（初回はダウンロード）... デバイス: {device}")
    if device == "cpu":
        print("  ※ GPU が使えないため、CPU で実行します（遅いです）")
    return whisper.load_model(MODEL_NAME, device=device), use_cuda


def transcribe_one(model, use_cuda: bool, path: Path, is_resp: bool) -> dict:
    return model.transcribe(
        str(path),
        language=LANGUAGE,
        word_timestamps=True,
        condition_on_previous_text=False,  # 前の文に引きずられた繰り返しを防ぐ
        initial_prompt=effective_prompt(is_resp),
        fp16=use_cuda,
    )


def filter_result(res: dict) -> tuple[str, list[dict], list[dict]]:
    """Whisper の結果から、幻覚らしい区間を取り除く。(本文, 残した区間, 除いた区間) を返す。"""
    all_segs = res.get("segments", [])
    dropped: list[dict] = []
    if not DROP_HALLUCINATIONS:
        return res.get("text", "").strip(), all_segs, dropped
    segs = []
    for sg in all_segs:
        t_ = sg.get("text", "")
        bad = (any(h in t_ for h in HALLUCINATIONS)
               or (sg.get("no_speech_prob", 0.0) > 0.6 and sg.get("avg_logprob", 0.0) < -1.0)
               or sg.get("compression_ratio", 0.0) > 2.4)
        (dropped if bad else segs).append(sg)
    return "".join(sg.get("text", "") for sg in segs).strip(), segs, dropped


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("ffmpeg が見つかりません。", file=sys.stderr)
        return 1
    csvs = sorted(ROLES_DIR.glob("*_roles.csv"))
    if not csvs:
        print(f"{ROLES_DIR} に *_roles.csv がありません（先に assign_roles.py を実行）", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    model = None

    for csv_path in csvs:
        base = csv_path.name[: -len("_roles.csv")]
        if TARGETS and base not in TARGETS:
            continue
        out_t, out_w = OUT_DIR / f"{base}_transcript.csv", OUT_DIR / f"{base}_words.csv"
        if SKIP_EXISTING and out_t.exists() and out_w.exists():
            print(f"[{base}] 出力済みのためスキップ")
            continue
        if model is None:
            model, use_cuda = load_model()

        rows = read_csv(csv_path)
        t_rows, w_rows = [], []
        t0 = time.time()
        for i, r in enumerate(rows, start=1):
            audio = SPLIT_DIR / base / r["ファイル名"]
            if not audio.exists():
                print(f"  [WARN] mp3がありません: {audio.name}")
                continue
            res = transcribe_one(model, use_cuda, audio, r["役割"] == "回答者")
            text, segs, dropped = filter_result(res)

            flags = []
            if dropped:
                flags.append("幻覚らしい区間を除去")
            if DROP_HALLUCINATIONS and not text:
                flags.append("Whisper空(Azureを使用)")
            if any(h in text for h in HALLUCINATIONS):
                flags.append("定型句")
            if only_fillers(text):
                flags.append("フィラーのみ")
            if segs and min(s.get("avg_logprob", 0.0) for s in segs) < -1.0:
                flags.append("低確信")
            if segs and max(s.get("no_speech_prob", 0.0) for s in segs) > 0.6:
                flags.append("無音の疑い")

            start_abs = parse_time(r["開始"]) - PAD_SEC
            for s in segs:
                for w in s.get("words", []):
                    w_rows.append({
                        "ファイル名": r["ファイル名"], "単語": w["word"].strip(),
                        "開始(相対秒)": f"{w['start']:.2f}", "終了(相対秒)": f"{w['end']:.2f}",
                        "開始(絶対秒)": f"{start_abs + w['start']:.2f}",
                        "終了(絶対秒)": f"{start_abs + w['end']:.2f}",
                        "確信度": f"{w.get('probability', 0.0):.2f}",
                    })
            t_rows.append({
                "通し番号": r["通し番号"], "話者": r["話者"], "役割": r["役割"], "開始": r["開始"], "終了": r["終了"],
                "ファイル名": r["ファイル名"], "文字起こし(Whisper)": text,
                "元の発言内容(Azure・名前伏せ済)": r["発話内容"], "要確認": " ".join(flags),
                "除去した区間": "｜".join(sg.get("text", "").strip() for sg in dropped),
                "フィラー数(Whisper)": count_fillers(text), "フィラー数(Azure)": count_fillers(r["発話内容"]),
            })
            if i % 10 == 0 or i == len(rows):
                print(f"  [{base}] {i}/{len(rows)}  経過 {time.time() - t0:.0f}秒")

        write_csv(out_t, t_rows, ["通し番号", "話者", "役割", "開始", "終了", "ファイル名",
                                  "文字起こし(Whisper)", "元の発言内容(Azure・名前伏せ済)", "要確認", "除去した区間",
                                  "フィラー数(Whisper)", "フィラー数(Azure)"])
        write_csv(out_w, w_rows, ["ファイル名", "単語", "開始(相対秒)", "終了(相対秒)",
                                  "開始(絶対秒)", "終了(絶対秒)", "確信度"])
        n_flag = sum(1 for t in t_rows if t["要確認"])
        fw = sum(t["フィラー数(Whisper)"] for t in t_rows)
        fa = sum(t["フィラー数(Azure)"] for t in t_rows)
        print(f"[{base}] {len(t_rows)}発話を文字起こし（要確認 {n_flag}件） / {time.time() - t0:.0f}秒"
              f" / フィラー数 Whisper {fw} : Azure {fa}")

    print(f"\n出力先: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())