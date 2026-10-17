#!/usr/bin/env python3
"""Train the MVIP soft prompt from configs/train.yaml."""

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
    parser = argparse.ArgumentParser(description="Train the MVIP soft prompt.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/train.yaml")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/train")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / ".cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--outer-steps",
        type=int,
        default=None,
        help="Override the configured outer-step count for a smoke test.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved training command and exit.")
    return parser.parse_args()


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an ablation override into its base configuration."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration is not a mapping: {path}")
    base_config = payload.pop("base_config", None)
    if base_config is None:
        return payload
    if not isinstance(base_config, str):
        raise ValueError(f"base_config must be a path string: {path}")
    base_path = (path.parent / base_config).resolve()
    return merge_config(load_yaml(base_path), payload)


def absolutize_data_paths(value: Any, data_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: absolutize_data_paths(item, data_root) for key, item in value.items()}
    if isinstance(value, list):
        return [absolutize_data_paths(item, data_root) for item in value]
    if isinstance(value, str) and value.startswith("data/"):
        return str(data_root / value.removeprefix("data/"))
    return value


def prepare_portable_inputs(config: dict[str, Any], data_root: Path, temp_root: Path) -> tuple[Path, Path]:
    temp_root.mkdir(parents=True, exist_ok=True)
    data = config["data"]
    source_manifest = REPO_ROOT / data["task_manifest"]
    source_schedule = REPO_ROOT / data["schedule_manifest"]

    manifest_payload = absolutize_data_paths(
        json.loads(source_manifest.read_text(encoding="utf-8")), data_root.resolve()
    )
    manifest_path = temp_root / "train_samples.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")

    schedule_payload = json.loads(source_schedule.read_text(encoding="utf-8"))
    outer_steps = schedule_payload.get("outer_steps", {})
    if sorted(int(step) for step in outer_steps) != list(range(1, 101)):
        raise ValueError(f"Expected exact outer steps 1-100 in {source_schedule}")

    schedule_root = temp_root / "schedule"
    schedule_state = schedule_root / "state"
    schedule_state.mkdir(parents=True, exist_ok=True)
    for step, task_batches in outer_steps.items():
        payload = absolutize_data_paths({"task_batches": task_batches}, data_root.resolve())
        destination = schedule_state / f"outer_step_{int(step):03d}_schedule.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path, schedule_root


def bool_flag(name: str, enabled: bool) -> str:
    return f"--{name}" if enabled else f"--no-{name}"


def build_runner_argv(
    config: dict[str, Any],
    args: argparse.Namespace,
    manifest_path: Path,
    schedule_root: Path,
) -> list[str]:
    vlm = config["vlm"]
    generator = config["generator"]
    optimization = config["optimization"]
    data = config["data"]

    init_mode = str(vlm.get("init_mode", "text")).lower()
    if init_mode not in {"text", "random"}:
        raise ValueError(f"Unsupported vlm.init_mode={init_mode!r}; expected 'text' or 'random'.")

    argv = [
        "mvip.train",
        "--task_manifest", str(manifest_path),
        "--prompt_path", str(REPO_ROOT / "mvip/prompts/initial_caption_prompt.txt"),
        "--n_soft_tokens", str(vlm["n_soft_tokens"]),
        "--out_root", str(args.output_dir.resolve()),
        "--run_name", str(config["experiment_name"]),
        "--vlm_model_id", str(vlm["model_id"]),
        "--device", str(args.device),
        "--cache_dir", str(args.cache_dir.resolve()),
        "--n_best_of", str(optimization["n_best_of"]),
        "--caption_temperature", str(vlm["caption_temperature"]),
        "--max_new_tokens", str(vlm["max_new_tokens"]),
        "--num_caption_frames", str(vlm["num_caption_frames"]),
        "--num_gen_frames", str(generator["num_frames"]),
        bool_flag("match_ref_num_frames", bool(generator["match_reference_num_frames"])),
        "--max_gen_seconds", str(generator["max_seconds"]),
        "--max_num_gen_frames", str(generator["max_num_frames"]),
        "--wan_conditioning", str(generator["conditioning"]),
        bool_flag("wan_cpu_offload", bool(generator["cpu_offload"])),
        "--num_tasks_per_outer_step", str(optimization["tasks_per_outer_step"]),
        "--support_size", str(optimization["support_size"]),
        "--min_frames_per_clip", str(data["min_frames_per_clip"]),
        "--outer_steps", str(
            args.outer_steps if args.outer_steps is not None else optimization["outer_steps"]
        ),
        "--inner_steps", str(optimization["inner_steps"]),
        "--inner_lr", str(optimization["inner_lr"]),
        "--epsilon", str(optimization["epsilon"]),
        "--seed", str(config["seed"]),
        "--schedule_from_run_root", str(schedule_root),
    ]
    if init_mode == "text":
        init_text = str(vlm.get("init_soft_prompt_text", "")).strip()
        if not init_text:
            raise ValueError("vlm.init_soft_prompt_text is required when vlm.init_mode is 'text'.")
        argv.extend(["--init_soft_prompt_text", init_text])
    if optimization.get("save_candidate_videos", False):
        argv.append("--save_score_videos")
    if args.resume:
        argv.append("--resume")
    return argv


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config.resolve())
    output_dir = args.output_dir.resolve()
    if args.dry_run:
        input_root_context = tempfile.TemporaryDirectory(prefix="mvip-train-")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        input_root_context = nullcontext(str(output_dir / "resolved_inputs"))

    with input_root_context as input_root:
        input_root_path = Path(input_root)
        input_root_path.mkdir(parents=True, exist_ok=True)
        manifest_path, schedule_root = prepare_portable_inputs(
            config,
            args.data_root,
            input_root_path,
        )
        runner_argv = build_runner_argv(config, args, manifest_path, schedule_root)
        print("Resolved command:\n" + shlex.join([sys.executable, "-m", "mvip.train", *runner_argv[1:]]))
        if args.dry_run:
            return
        os.chdir(REPO_ROOT)
        sys.argv = runner_argv
        from mvip.train import main as runner_main

        runner_main()


if __name__ == "__main__":
    main()
