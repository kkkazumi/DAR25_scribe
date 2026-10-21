#!/usr/bin/env python3
"""
pyannote で mp3 から「2人が同時に話している区間（重なり）」を検出するスクリプト（PyCharm用）。

使い方:
    1. 下の【設定】を確認する（TARGETS だけ変えれば基本OK）
    2. PyCharm でこのファイルを開き、右クリック → Run 'detect_overlap'

事前準備:
    pip install pyannote.audio          （PyCharm の Terminal タブで実行）
    Hugging Face で次の2ページの利用条件に同意し、アクセストークンを発行:
        https://huggingface.co/pyannote/speaker-diarization-3.1
        https://huggingface.co/pyannote/segmentation-3.0
    トークンの渡し方（どれか1つ）:
        a) このファイルと同じ階層に hf_token.txt を作り、トークンだけを書く（おすすめ）
        b) 環境変数 HF_TOKEN に設定
        c) ターミナルで一度 huggingface-cli login しておく
    ffmpeg が PATH に通っていること（PATH を変えた直後は PyCharm を再起動）

出力（out_overlap/<mp3名>/）:
    turns.csv              pyannote が判定した話者ごとの発話区間
    overlaps.csv           2人以上が同時に話している区間
    utterance_overlap.csv  csv_merged に対応するCSVがあれば、各発話の「重なり率」を付けたもの
"""
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# ============== 【設定】 ==============
# mp3/ フォルダ内で処理するファイル名。空リスト [] にすると mp3/ 内のすべてを処理する
TARGETS = []
NUM_SPEAKERS = 2                              # 話者数（インタビューなので2）
MODEL = "pyannote/speaker-diarization-3.1"
FLAG_RATIO = 0.10                             # 重なり率がこれ以上の発話に「要確認」(●)を付ける
# =====================================

BASE = Path(__file__).resolve().parent
MP3_DIR = BASE / "mp3"
MERGED_DIR = BASE / "csv_merged"
OUT_DIR = BASE / "out_overlap"


def get_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    f = BASE / "hf_token.txt"
    if f.exists():
        return f.read_text(encoding="utf-8").strip()
    return None  # huggingface-cli login 済みなら、None でも保存済みトークンが使われる


def parse_time(s: str) -> float:
    sec = 0.0
    for part in s.strip().split(":"):
        sec = sec * 60 + float(part)
    return sec


def fmt_time(sec: float) -> str:
    return f"{int(sec) // 3600:02d}:{int(sec) % 3600 // 60:02d}:{sec % 60:05.2f}"


def overlap_ratio(start: float, end: float, overlaps: list[tuple[float, float]]) -> float:
    """[start, end] のうち、重なり区間が占める割合"""
    dur = end - start
    if dur <= 0:
        return 0.0
    total = 0.0
    for s, e in overlaps:
        total += max(0.0, min(end, e) - max(start, s))
    return min(1.0, total / dur)


def to_wav(src: Path, dst: Path) -> None:
    """mp3 → 16kHz モノラル wav（pyannote が確実に読める形式）"""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip())


def load_wav(path: Path) -> dict:
    """16bit モノラル wav を読み込み、pyannote にメモリごと渡せる形にする。
    （pyannote 4.x がファイル読み込みに使う torchcodec / FFmpeg DLL を回避するため）"""
    import wave

    import numpy as np
    import torch

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    waveform = torch.from_numpy(data.astype(np.float32) / 32768.0).unsqueeze(0)  # (1, samples)
    return {"waveform": waveform, "sample_rate": sr}


def load_pipeline(model: str, token: str | None):
    import torch
    from pyannote.audio import Pipeline

    try:
        pipe = Pipeline.from_pretrained(model, token=token)
    except TypeError:  # 古い pyannote.audio
        pipe = Pipeline.from_pretrained(model, use_auth_token=token)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda"))
        print("GPU を使用します")
    else:
        print("GPU が見つかりません。CPU で実行します（時間がかかります）")
    return pipe


def find_merged_csv(stem: str) -> Path | None:
    """csv_merged 内で、mp3名で始まる *_merged.csv を探す"""
    cands = sorted(MERGED_DIR.glob(f"{stem}*_merged.csv"))
    return cands[0] if cands else None


def process(mp3: Path, pipe) -> None:
    out_dir = OUT_DIR / mp3.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "audio.wav"
        print(f"[{mp3.name}] wav に変換中...")
        to_wav(mp3, wav)
        print(f"[{mp3.name}] 話者分離を実行中...（CPUだと数分〜十数分かかります）")
        audio = load_wav(wav)
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            with ProgressHook() as hook:  # 進捗を表示
                out = pipe(audio, num_speakers=NUM_SPEAKERS, hook=hook)
        except ImportError:
            out = pipe(audio, num_speakers=NUM_SPEAKERS)
    # pyannote.audio 4.x は結果オブジェクト、3.x は Annotation を直接返す
    diar = out.speaker_diarization if hasattr(out, "speaker_diarization") else out

    # 話者ごとの発話区間
    with (out_dir / "turns.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pyannote話者", "開始(秒)", "終了(秒)", "開始", "終了"])
        for seg, _, label in diar.itertracks(yield_label=True):
            w.writerow([label, f"{seg.start:.2f}", f"{seg.end:.2f}", fmt_time(seg.start), fmt_time(seg.end)])

    # 重なり区間（2人以上が同時に話している時間帯）
    overlaps = [(s.start, s.end) for s in diar.get_overlap()]
    with (out_dir / "overlaps.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["開始(秒)", "終了(秒)", "長さ(秒)", "開始", "終了"])
        for s, e in overlaps:
            w.writerow([f"{s:.2f}", f"{e:.2f}", f"{e - s:.2f}", fmt_time(s), fmt_time(e)])

    total_ov = sum(e - s for s, e in overlaps)
    total_dur = max((seg.end for seg in diar.itersegments()), default=0.0)
    print(f"\n[{mp3.name}] 重なり区間: {len(overlaps)}個 / 合計 {total_ov:.1f}秒"
          f"（音声全体 {total_dur:.0f}秒の {100 * total_ov / total_dur if total_dur else 0:.1f}%）")
    for s, e in sorted(overlaps, key=lambda x: x[0] - x[1])[:10]:
        print(f"  長い順: {fmt_time(s)} - {fmt_time(e)}（{e - s:.2f}秒）")

    # 各発話の重なり率（対応する *_merged.csv があれば）
    csv_path = find_merged_csv(mp3.stem)
    if csv_path is None:
        print(f"  （csv_merged に {mp3.stem} 対応のCSVがないため、発話ごとの重なり率はスキップ）")
    else:
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        n_flag = 0
        with (out_dir / "utterance_overlap.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(rows[0].keys()) + ["重なり率", "要確認"])
            for r in rows:
                ratio = overlap_ratio(parse_time(r["開始"]), parse_time(r["終了"]), overlaps)
                flag = "●" if ratio >= FLAG_RATIO else ""
                n_flag += bool(flag)
                w.writerow(list(r.values()) + [f"{ratio:.2f}", flag])
        print(f"  重なり率 {FLAG_RATIO:.0%} 以上の発話: {n_flag} / {len(rows)}件（{csv_path.name}）")
    print(f"  出力先: {out_dir}\n")


def main() -> int:
    names = sys.argv[1:] or TARGETS  # コマンドラインで名前を渡すことも可能
    mp3s = [MP3_DIR / n for n in names] if names else sorted(MP3_DIR.glob("*.mp3"))
    if not mp3s:
        print(f"mp3 が見つかりません: {MP3_DIR}", file=sys.stderr)
        return 1
    missing = [p for p in mp3s if not p.exists()]
    if missing:
        print("見つからないmp3:", ", ".join(p.name for p in missing), file=sys.stderr)
        return 1

    print("モデルを読み込み中（初回はダウンロードがあります）...")
    pipe = load_pipeline(MODEL, get_token())
    for mp3 in mp3s:
        process(mp3, pipe)
    return 0


if __name__ == "__main__":
    sys.exit(main())