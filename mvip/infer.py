#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mvip.video_io import (  # noqa: E402
    CAPTION_INSTRUCTION,
    DEFAULT_NEGATIVE_PROMPT,
    ManifestItem,
    extract_boundary_frames,
    sample_frames,
    save_reference_chunk,
)

OUTPUT_KEYS = ("output_video", "reference_video", "first_frame", "last_frame", "caption_path", "metadata_path")
DEFAULT_CONFIG_PATH = REPO_ROOT / "external" / "VideoX-Fun" / "config" / "wan2.2" / "wan_civitai_5b.yaml"
DEFAULT_MODEL_NAME = REPO_ROOT / "external" / "Wan2.2-Fun-5B-InP"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task_manifest", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--soft_tokens_path", type=Path, default=None, help="Required for --caption_mode soft_prompt.")
    parser.add_argument(
        "--caption_mode",
        choices=["soft_prompt", "captioning", "flf_only"],
        default="soft_prompt",
        help="soft_prompt: MVIP (VLM + learned soft prompt); captioning: plain VLM caption baseline; "
        "flf_only: empty prompt, first/last-frame conditioning only.",
    )
    parser.add_argument("--prompt_text", default=CAPTION_INSTRUCTION)
    parser.add_argument("--vlm_model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cache_dir", type=Path, default=REPO_ROOT / ".cache")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_caption_frames", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42, help="Used only for items without a seed field.")
    # VideoX-Fun generation.
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model_name", type=Path, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--sample_size", type=int, nargs=2, default=[704, 1280])
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--gpu_memory_mode", default="model_full_load")
    parser.add_argument("--sampler_name", default="Flow")
    parser.add_argument("--shift", type=float, default=5.0)
    # Scheduling.
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Write the remapped manifest and plan only.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def remap_item(raw: dict[str, Any], out_dir: Path, default_seed: int, default_fps: int) -> ManifestItem:
    item = dict(raw)
    for key in OUTPUT_KEYS:
        old = Path(item[key])
        parts = old.parts
        if "videos" in parts:
            rel = Path(*parts[parts.index("videos"):])
        else:
            rel = Path("videos") / item["source"] / old.parent.parent.name / old.parent.name / old.name
        item[key] = str(out_dir / rel)
    item["caption"] = ""
    item.setdefault("seed", default_seed)
    item.setdefault("fps", default_fps)
    item.setdefault("chunk_index", 0)
    item.setdefault("input_fps", None)
    item.setdefault("input_frame_count", None)
    allowed = {f.name for f in fields(ManifestItem)}
    return ManifestItem(**{k: v for k, v in item.items() if k in allowed})




def load_qwen(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model_id,
        torch_dtype=dtype,
        device_map="auto",
        cache_dir=str(args.cache_dir),
    )
    processor = AutoProcessor.from_pretrained(args.vlm_model_id, cache_dir=str(args.cache_dir))
    processor.tokenizer.padding_side = "left"
    model.requires_grad_(False)
    return model, processor


def generator_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "backend": "videox_fun_flf",
        "config_path": str(args.config_path.resolve()),
        "model_name": str(args.model_name.resolve()),
        "sample_size": [int(v) for v in args.sample_size],
        "fps": args.fps,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "negative_prompt": args.negative_prompt,
        "gpu_memory_mode": args.gpu_memory_mode,
        "sampler_name": args.sampler_name,
        "shift": args.shift,
    }




def run_generation(args: argparse.Namespace, items: list[ManifestItem], gen_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    from mvip.utils import generate_caption_with_phi, qwen_caption
    from mvip.wan_flf import build_pipeline, generate_segment, make_i2v_args

    device = args.device if torch.cuda.is_available() else "cpu"
    phi = None
    if args.caption_mode == "soft_prompt":
        phi = torch.load(args.soft_tokens_path, map_location=device)
        print(
            "[qwen-input] "
            f"soft_tokens={args.soft_tokens_path} shape={tuple(phi.shape)} "
            f"norm={float(phi.float().norm().item()):.6f} prompt={args.prompt_text!r}",
            flush=True,
        )
    print(f"[phase] caption pass start (mode={args.caption_mode})", flush=True)
    qwen = qwen_proc = None
    if args.caption_mode != "flf_only":
        qwen, qwen_proc = load_qwen(args)

    captioned: list[ManifestItem] = []
    for idx, item in enumerate(items, start=1):
        out_path = Path(item.output_video)
        caption_path = Path(item.caption_path)
        if args.skip_existing and out_path.exists() and caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
            captioned.append(ManifestItem(**{**asdict(item), "caption": caption}))
            print(f"[qwen-skip] {idx}/{len(items)} existing generated+caption {item.source}/{item.sample_id}", flush=True)
            continue

        print(f"[qwen-run] {idx}/{len(items)} {item.source}/{item.sample_id}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_reference_chunk(item.input_video, item.chunk_start_sec, item.chunk_duration_sec, Path(item.reference_video))
        extract_boundary_frames(
            Path(item.input_video),
            item.chunk_start_sec,
            item.chunk_end_sec,
            item.input_fps,
            item.input_frame_count,
            Path(item.first_frame),
            Path(item.last_frame),
        )
        if args.caption_mode == "flf_only":
            caption = ""
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            caption_path.write_text("\n", encoding="utf-8")
            captioned.append(ManifestItem(**{**asdict(item), "caption": caption}))
            continue
        frames = sample_frames(Path(item.reference_video), args.num_caption_frames)
        print(
            "[qwen-input] "
            f"item={idx}/{len(items)} source={item.source} sample_id={item.sample_id} "
            f"reference={item.reference_video} frames={len(frames)} "
            f"sizes={[f'{f.width}x{f.height}' for f in frames]} n_soft={0 if phi is None else phi.shape[0]}",
            flush=True,
        )
        if args.skip_existing and caption_path.exists():
            caption = caption_path.read_text(encoding="utf-8").strip()
            print(f"[qwen-skip] caption exists: {caption_path}", flush=True)
        elif args.caption_mode == "captioning":
            caption = qwen_caption(qwen, qwen_proc, frames, args.prompt_text, args.max_new_tokens)
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            caption_path.write_text(caption + "\n", encoding="utf-8")
            print(f"[qwen-output] item={idx}/{len(items)} caption_chars={len(caption)} caption={caption[:180]!r}", flush=True)
        else:
            caption = generate_caption_with_phi(
                vlm_model=qwen,
                vlm_processor=qwen_proc,
                frames=frames,
                prompt_text=args.prompt_text,
                soft_tokens=phi,
                max_new_tokens=args.max_new_tokens,
            )
            caption_path.parent.mkdir(parents=True, exist_ok=True)
            caption_path.write_text(caption + "\n", encoding="utf-8")
            print(f"[qwen-output] item={idx}/{len(items)} caption_chars={len(caption)} caption={caption[:180]!r}", flush=True)
        captioned.append(ManifestItem(**{**asdict(item), "caption": caption}))

    print("[phase] caption pass complete; unloading the VLM", flush=True)
    del qwen, qwen_proc, phi
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    rows: list[dict[str, Any]] = []
    summary_path = args.out_dir / "flf_eval_summary.json"
    pending = [item for item in captioned if not (args.skip_existing and Path(item.output_video).exists())]
    pipeline_bundle = None
    if pending:
        print("[phase] wan generation pass start; loading VideoX-Fun pipeline once", flush=True)
        pipeline_bundle = build_pipeline(make_i2v_args(gen_cfg, pending[0].video_length))

    for idx, item in enumerate(captioned, start=1):
        out_path = Path(item.output_video)
        row = asdict(item)
        row["caption_mode"] = args.caption_mode
        row["soft_tokens_path"] = str(args.soft_tokens_path.resolve()) if args.soft_tokens_path else None
        row["prompt_text"] = args.prompt_text
        row["generator"] = gen_cfg
        if args.skip_existing and out_path.exists():
            row["status"] = "skipped"
            rows.append(row)
            print(f"[wan-skip] {idx}/{len(captioned)} existing generated {item.source}/{item.sample_id}", flush=True)
            continue

        pipeline, wan_device, boundary = pipeline_bundle
        caption = item.caption if args.caption_mode == "flf_only" else (item.caption or Path(item.caption_path).read_text(encoding="utf-8").strip())
        print(
            f"[wan-run] {idx}/{len(captioned)} {item.source}/{item.sample_id} "
            f"video_length={item.video_length} first={item.first_frame} last={item.last_frame}",
            flush=True,
        )
        generate_segment(
            make_i2v_args(gen_cfg, item.video_length),
            pipeline,
            pipeline.vae,
            wan_device,
            boundary,
            caption,
            item.first_frame,
            item.last_frame,
            out_path,
            item.seed,
        )
        row["status"] = "done"
        write_json(Path(item.metadata_path), row)
        rows.append(row)
        write_json(summary_path, {"num_items": len(captioned), "rows": rows})
        print(f"[done] {idx}/{len(captioned)} {item.sample_id}", flush=True)

    write_json(summary_path, {"num_items": len(captioned), "rows": rows})
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    if args.caption_mode == "soft_prompt":
        if args.soft_tokens_path is None or not args.soft_tokens_path.is_file():
            raise FileNotFoundError(f"--soft_tokens_path is required for --caption_mode soft_prompt: {args.soft_tokens_path}")
    payload = read_json(args.task_manifest)
    raw_items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError(f"Manifest has no item list: {args.task_manifest}")

    items = [remap_item(raw, args.out_dir, args.seed, args.fps) for raw in raw_items]
    seed = payload.get("seed", args.seed) if isinstance(payload, dict) else args.seed
    write_json(
        args.out_dir / f"manifest_seed{seed}.json",
        {
            "baseline": {"soft_prompt": "mvip", "captioning": "captioning", "flf_only": "flf-only"}[args.caption_mode],
            "seed": seed,
            "caption_instruction": args.prompt_text,
            "num_items": len(items),
            "items": [asdict(item) for item in items],
        },
    )
    run_items = items[: args.limit] if args.limit > 0 else items
    gen_cfg = generator_config(args)
    write_json(
        args.out_dir / "eval_config.json",
        {
            "task_manifest": str(args.task_manifest.resolve()),
            "caption_mode": args.caption_mode,
            "soft_tokens_path": str(args.soft_tokens_path.resolve()) if args.soft_tokens_path else None,
            "prompt_text": args.prompt_text,
            "vlm_model_id": args.vlm_model_id,
            "num_caption_frames": args.num_caption_frames,
            "max_new_tokens": args.max_new_tokens,
            "generator": gen_cfg,
            "num_items": len(run_items),
        },
    )
    print(f"[*] items={len(run_items)} out_dir={args.out_dir}", flush=True)
    for item in run_items:
        print(f"    {item.source}/{item.sample_id} video_length={item.video_length} -> {item.output_video}")
    if args.dry_run:
        return
    run_generation(args, run_items, gen_cfg)
    print(f"[*] wrote FLF eval outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
