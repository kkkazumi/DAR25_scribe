#!/usr/bin/env python3
"""
CSV（話者・開始・終了・発話内容）に基づいて mp3 を発話ごとに分割する。

ディレクトリ構成（このスクリプトと同じ階層）:
    csv/   元のCSV（例: 001_DAR25_001_Human.csv）
    mp3/   元のmp3（例: 001_DAR25_001_Human.mp3） ※CSVと拡張子以外同名
出力:
    csv_merged/<stem>_merged.csv   連続同一話者をマージし通し番号を付けたCSV
    out_mp3/<stem>/<stem>_<通し番号>_spk<話者番号>.mp3

手順:
    1. CSVを読み込む（元ファイルは変更しない。結果は別名で保存）
    2. 同一話者が連続する行は 開始=最初の開始 / 終了=最後の終了 にマージ
    3. 1行目から順に通し番号（話者間で共通）
    4. 開始・終了時刻でmp3を切り出し、話者番号＋通し番号をファイル名に付ける

必要なもの: ffmpeg（PATHが通っていること）
使い方:
    python split_audio.py            # 実行
    python split_audio.py --dry-run  # CSVのマージだけ行い、mp3は切り出さない
    python split_audio.py --reencode # 再エンコードで切り出す（遅いが境界が正確）
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV_DIR = BASE / "csv"
MP3_DIR = BASE / "mp3"
MERGED_DIR = BASE / "csv_merged"
OUT_DIR = BASE / "out_mp3"

PAD_SEC = 0.0  # 前後に足す余白（秒）。必要なら 0.2 などに

COL_SPK, COL_START, COL_END, COL_TEXT = "話者", "開始", "終了", "発話内容"


def parse_time(s: str) -> float:
    """'HH:MM:SS' / 'MM:SS' / 'H:MM:SS.mmm' / 秒数 のいずれかを秒に変換"""
    s = s.strip()
    if ":" not in s:
        return float(s)
    sec = 0.0
    for part in s.split(":"):
        sec = sec * 60 + float(part)
    return sec


def fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def read_rows(path: Path) -> list[dict]:
    """文字コード（UTF-8 / CP932）と区切り文字（カンマ / タブ）を自動判定して読む"""
    text = None
    for enc in ("utf-8-sig", "cp932"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"文字コードを判定できません: {path}")

    first_line = text.splitlines()[0]
    delim = "\t" if "\t" in first_line else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    reader.fieldnames = [(n or "").strip() for n in reader.fieldnames]

    missing = [c for c in (COL_SPK, COL_START, COL_END, COL_TEXT) if c not in reader.fieldnames]
    if missing:
        raise ValueError(f"{path.name}: 列が見つかりません {missing} / 実際の列: {reader.fieldnames}")

    rows = []
    for r in reader:
        if not (r.get(COL_SPK) or "").strip():
            continue
        m = re.search(r"\d+", r[COL_SPK])
        if not m:
            raise ValueError(f"{path.name}: 話者番号を読み取れません: {r[COL_SPK]!r}")
        rows.append({
            "spk": int(m.group()),
            "start": parse_time(r[COL_START]),
            "end": parse_time(r[COL_END]),
            "text": (r.get(COL_TEXT) or "").strip(),
        })
    return rows


def merge_consecutive(rows: list[dict]) -> list[dict]:
    """同一話者が連続する行を1つにまとめる"""
    merged: list[dict] = []
    for r in rows:
        if merged and merged[-1]["spk"] == r["spk"]:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
            merged[-1]["text"] += r["text"]  # 日本語なので区切りなしで連結
        else:
            merged.append(dict(r))
    return merged


def find_mp3(csv_stem: str) -> Path | None:
    """CSV名に対応するmp3を探す。
    1) 同名のmp3  2) mp3名がCSV名の先頭部分と一致するもの
    （例: DAR25_001_Human_transcript.csv ↔ DAR25_001_Human.mp3）"""
    exact = MP3_DIR / f"{csv_stem}.mp3"
    if exact.exists():
        return exact
    cands = [p for p in MP3_DIR.glob("*.mp3") if csv_stem.startswith(p.stem)]
    return max(cands, key=lambda p: len(p.stem)) if cands else None


def cut_mp3(src: Path, dst: Path, start: float, end: float, reencode: bool) -> None:
    start = max(0.0, start - PAD_SEC)
    dur = (end + PAD_SEC) - start
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}"]
    cmd += ["-c:a", "libmp3lame", "-q:a", "2"] if reencode else ["-c", "copy"]
    cmd.append(str(dst))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="mp3は切り出さず、マージ済みCSVのみ作成")
    ap.add_argument("--reencode", action="store_true", help="再エンコードして切り出す")
    args = ap.parse_args()

    if not args.dry_run and shutil.which("ffmpeg") is None:
        print("ffmpeg が見つかりません。インストールして PATH を通してください。", file=sys.stderr)
        return 1

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    if not csv_files:
        print(f"CSVが見つかりません: {CSV_DIR}", file=sys.stderr)
        return 1

    MERGED_DIR.mkdir(exist_ok=True)
    failed = []

    for csv_path in csv_files:
        stem = csv_path.stem
        try:
            rows = read_rows(csv_path)
        except ValueError as e:
            print(f"[SKIP] {e}")
            failed.append(stem)
            continue

        mp3_path = find_mp3(stem)
        base = mp3_path.stem if mp3_path else stem  # 出力ファイル名のベース（mp3名を優先）

        merged = merge_consecutive(rows)
        width = max(3, len(str(len(merged))))
        for i, r in enumerate(merged, start=1):
            r["seq"] = i
            r["name"] = f"{base}_{i:0{width}d}_spk{r['spk']}.mp3"

        # マージ済みCSVを別名で保存（元ファイルは触らない）
        merged_csv = MERGED_DIR / f"{stem}_merged.csv"
        with merged_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["通し番号", COL_SPK, COL_START, COL_END, COL_TEXT, "ファイル名"])
            for r in merged:
                w.writerow([r["seq"], f"話者{r['spk']}", fmt_time(r["start"]),
                            fmt_time(r["end"]), r["text"], r["name"]])

        msg = f"{stem}: {len(rows)}行 → {len(merged)}行"
        if args.dry_run:
            print(msg + "（dry-run）")
            continue

        if mp3_path is None:
            print(f"[SKIP] {msg} / 対応するmp3が mp3/ に見つかりません")
            failed.append(stem)
            continue

        out_dir = OUT_DIR / base
        out_dir.mkdir(parents=True, exist_ok=True)
        n_ok = 0
        for r in merged:
            if r["end"] <= r["start"]:
                print(f"  [WARN] {r['name']}: 終了<=開始のためスキップ")
                continue
            try:
                cut_mp3(mp3_path, out_dir / r["name"], r["start"], r["end"], args.reencode)
                n_ok += 1
            except RuntimeError as e:
                print(f"  [ERROR] {r['name']}: {e}")
        print(f"{msg} / mp3 {n_ok}個を出力")

    if failed:
        print("\n処理できなかったもの:", ", ".join(failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())