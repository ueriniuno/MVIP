#!/usr/bin/env python3
"""Run MVIP inference with the released soft prompt on one video or a manifest."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MVIP with the released soft prompt.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-video", type=Path)
    inputs.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/eval.yaml")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline", choices=["captioning", "flf-only"], default=None,
                        help="Run a paper baseline instead of MVIP: plain VLM captioning, or first/last frames with an empty prompt.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/inference")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_manifest_paths(value: Any, data_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: resolve_manifest_paths(item, data_root) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_manifest_paths(item, data_root) for item in value]
    if isinstance(value, str) and value.startswith("data/"):
        return str(data_root / value.removeprefix("data/"))
    return value


def single_video_item(video: Path, seed: int, gen_cfg: dict[str, Any]) -> dict[str, Any]:
    """Manifest entry for one user video."""
    from mvip.video_io import (
        chunk_label,
        get_video_info,
        video_length_for_duration,
    )

    duration, input_fps, frame_count = get_video_info(video)
    fps = int(gen_cfg.get("fps", 24))
    video_length = video_length_for_duration(duration, fps, int(gen_cfg.get("user_video_fallback_length", 121)))
    max_length = int(gen_cfg.get("user_video_max_length", 0) or 0)
    if max_length and video_length > max_length:
        video_length = max_length - ((max_length - 1) % 4)
    label = chunk_label(0, 0.0, duration)
    rel = Path("videos") / "user" / video.stem / label
    return {
        "source": "user",
        "sample_id": f"{video.stem}_{label}",
        "input_video": str(video),
        "chunk_start_sec": 0.0,
        "chunk_end_sec": duration,
        "chunk_duration_sec": duration,
        "chunk_index": 0,
        "output_video": str(rel / "generated.mp4"),
        "reference_video": str(rel / "reference.mp4"),
        "first_frame": str(rel / "first_frame.png"),
        "last_frame": str(rel / "last_frame.png"),
        "caption_path": str(rel / "caption.txt"),
        "metadata_path": str(rel / "metadata.json"),
        "caption": "",
        "seed": seed,
        "video_length": video_length,
        "fps": fps,
        "input_fps": input_fps,
        "input_frame_count": frame_count,
    }


def make_manifest(args: argparse.Namespace, config: dict[str, Any], gen_cfg: dict[str, Any], destination: Path) -> Path:
    instruction = str(config["vlm"]["instruction"])
    if args.input_video is not None:
        video = args.input_video.resolve()
        if not video.is_file():
            raise FileNotFoundError(video)
        item = single_video_item(video, int(config["seed"]), gen_cfg)
        payload = {"caption_instruction": instruction, "seed": int(config["seed"]), "num_items": 1, "items": [item]}
    else:
        source = (args.manifest or REPO_ROOT / config["manifest"]).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload = resolve_manifest_paths(payload, args.data_root.resolve())
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return destination


def resolve_generator_paths(gen_cfg: dict[str, Any]) -> tuple[Path, Path]:
    """VideoX-Fun checkout and weights from the config, overridable by environment variables."""
    configured_videox_root = REPO_ROOT / "external" / "VideoX-Fun"
    videox_root = Path(os.environ.get("MVIP_VIDEOX_FUN_ROOT", configured_videox_root)).resolve()
    model_root = Path(os.environ.get("MVIP_WAN_FUN_MODEL_ROOT", REPO_ROOT / gen_cfg["model_name"])).resolve()
    configured_config_path = (REPO_ROOT / gen_cfg["config_path"]).resolve()
    try:
        config_path = videox_root / configured_config_path.relative_to(configured_videox_root.resolve())
    except ValueError:
        config_path = configured_config_path
    return config_path, model_root


def runner_argv(args: argparse.Namespace, config: dict[str, Any], gen_cfg: dict[str, Any], manifest: Path, checkpoint: Path, output_dir: Path) -> list[str]:
    vlm = config["vlm"]
    config_path, model_root = resolve_generator_paths(gen_cfg)
    argv = [
        "mvip.infer",
        "--task_manifest", str(manifest),
        "--out_dir", str(output_dir),
        "--soft_tokens_path", str(checkpoint),
        "--caption_mode", {None: "soft_prompt", "captioning": "captioning", "flf-only": "flf_only"}[args.baseline],
        "--prompt_text", str(vlm["instruction"]),
        "--vlm_model_id", str(vlm["model_id"]),
        "--cache_dir", str(args.cache_dir.resolve()),
        "--device", str(args.device),
        "--num_caption_frames", str(vlm["num_image_frames"]),
        "--max_new_tokens", str(vlm["max_new_tokens"]),
        "--seed", str(config["seed"]),
        "--config_path", str(config_path),
        "--model_name", str(model_root),
        "--fps", str(gen_cfg["fps"]),
        "--sample_size", str(gen_cfg["sample_size"][0]), str(gen_cfg["sample_size"][1]),
        "--guidance_scale", str(gen_cfg["guidance_scale"]),
        "--num_inference_steps", str(gen_cfg["num_inference_steps"]),
        "--negative_prompt", str(gen_cfg["negative_prompt"]),
        "--gpu_memory_mode", str(gen_cfg["gpu_memory_mode"]),
        "--sampler_name", str(gen_cfg["sampler_name"]),
        "--shift", str(gen_cfg["shift"]),
        "--limit", str(args.limit),
    ]
    if not args.overwrite:
        argv.append("--skip_existing")
    return argv


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    gen_cfg = config["generator"]
    checkpoint = (args.checkpoint or REPO_ROOT / config["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    output_dir = args.output_dir.resolve()
    if args.dry_run:
        manifest_root_context = tempfile.TemporaryDirectory(prefix="mvip-infer-")
        manifest_name = "manifest.json"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_root_context = nullcontext(str(output_dir))
        manifest_name = "input_manifest.json"

    with manifest_root_context as manifest_root:
        manifest = make_manifest(args, config, gen_cfg, Path(manifest_root) / manifest_name)
        argv = runner_argv(args, config, gen_cfg, manifest, checkpoint, output_dir)
        module = "mvip.infer"
        print("Resolved command:\n" + shlex.join([sys.executable, "-m", module, *argv[1:]]))
        if args.dry_run:
            print(Path(manifest).read_text(encoding="utf-8")[:1200])
            return
        os.chdir(REPO_ROOT)
        sys.argv = argv
        from mvip.infer import main as run

        run()


if __name__ == "__main__":
    main()
