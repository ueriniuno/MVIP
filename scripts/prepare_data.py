#!/usr/bin/env python3
"""Derive the evaluation/training inputs that are not part of the raw datasets:
scene segments of the ActivityNet / MSR-VTT videos and Wan2.2-TI2V-5B clips
generated from VidProM prompts (see README, "Data")."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFESTS = ("manifests/eval_138.json", "manifests/train_samples.json")
SEGMENTS = REPO_ROOT / "manifests/eval_segments.json"
# ActivityNet segments were cut with a stream copy, MSR-VTT segments were re-encoded.
CUT_MODE = {"ActivityNet": "copy", "MSR-VTT": "reencode"}
WAN_VIDEO_RE = re.compile(r"data/(VidProM/wan_(81|121)f/\d{5}_([0-9a-f-]{36})/seed_42\.mp4)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--segments", action="store_true", help="Only cut the ActivityNet/MSR-VTT segments.")
    parser.add_argument("--vidprom-wan", action="store_true", help="Only generate the Wan2.2-TI2V-5B clips.")
    parser.add_argument("--vidprom-csv", type=Path, default=None,
                        help="VidProM_unique.csv (default: <data-root>/VidProM/VidProM_unique.csv).")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".cache")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (args.segments or args.vidprom_wan):
        args.segments = args.vidprom_wan = True
    return args


def cut_segments(data_root: Path) -> None:
    from mvip.video_io import ffmpeg_exe

    spec = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    for dataset, videos in spec.items():
        for video_id, spans in videos.items():
            source = data_root / dataset / "videos" / f"{video_id}.mp4"
            if not source.is_file():
                raise FileNotFoundError(source)
            out_dir = data_root / dataset / "segments" / video_id
            out_dir.mkdir(parents=True, exist_ok=True)
            for index, (start, end) in enumerate(spans):
                out_path = out_dir / f"seg_{index:03d}.mp4"
                if out_path.exists():
                    continue
                if CUT_MODE[dataset] == "copy":
                    command = ["-ss", str(start), "-to", str(end), "-i", str(source),
                               "-c", "copy", "-avoid_negative_ts", "make_zero"]
                else:
                    command = ["-i", str(source), "-ss", f"{start:.3f}", "-t", f"{max(0.001, end - start):.3f}",
                               "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
                               "-crf", "18", "-c:a", "aac", "-movflags", "+faststart"]
                subprocess.run([ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *command, str(out_path)],
                               check=True)
            print(f"[segments] {dataset}/{video_id}: {len(spans)} segments", flush=True)


def wan_targets(data_root: Path) -> dict[Path, tuple[str, int]]:
    targets: dict[Path, tuple[str, int]] = {}
    for rel in MANIFESTS:
        for item in json.loads((REPO_ROOT / rel).read_text(encoding="utf-8"))["items"]:
            match = WAN_VIDEO_RE.fullmatch(item["input_video"])
            if match:
                targets[data_root / match.group(1)] = (match.group(3), int(match.group(2)))
    return targets


def generate_wan_clips(data_root: Path, csv_path: Path, cache_dir: Path, seed: int) -> None:
    from mvip.wan_diffusers import FPS, generate_video_frames, load_wan_pipeline, save_video

    targets = {path: spec for path, spec in wan_targets(data_root).items() if not path.exists()}
    if not targets:
        print("[vidprom-wan] all clips present", flush=True)
        return
    wanted = {uuid for uuid, _ in targets.values()}
    prompts: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["uuid"] in wanted:
                prompts[row["uuid"]] = row["prompt"].strip()
    missing = sorted(wanted - set(prompts))
    if missing:
        raise KeyError(f"prompts not found in {csv_path}: {missing}")

    pipe, device = load_wan_pipeline(cache_dir=str(cache_dir))
    for index, (path, (uuid, num_frames)) in enumerate(sorted(targets.items()), start=1):
        frames = generate_video_frames(pipe, device, prompts[uuid], seed, num_frames)
        save_video(frames, path, FPS)
        print(f"[vidprom-wan] {index}/{len(targets)} {path.relative_to(data_root)}", flush=True)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if args.segments:
        cut_segments(data_root)
    if args.vidprom_wan:
        csv_path = (args.vidprom_csv or data_root / "VidProM" / "VidProM_unique.csv").resolve()
        generate_wan_clips(data_root, csv_path, args.cache_dir.resolve(), args.seed)


if __name__ == "__main__":
    main()
