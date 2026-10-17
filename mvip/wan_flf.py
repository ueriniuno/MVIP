"""Evaluation generator: VideoX-Fun Wan2.2-Fun-5B-InP with first/last-frame conditioning."""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEOX_FUN_ROOT = REPO_ROOT / "external" / "VideoX-Fun"


def videox_fun_root() -> Path:
    return Path(os.environ.get("MVIP_VIDEOX_FUN_ROOT", DEFAULT_VIDEOX_FUN_ROOT)).resolve()


def ensure_sys_path():
    root = videox_fun_root()
    if not (root / "videox_fun").is_dir():
        raise FileNotFoundError(
            f"VideoX-Fun checkout not found at {root}. See the README "
            "(clone aigc-apps/VideoX-Fun and apply external/videox_fun.patch)."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def make_i2v_args(cfg: dict, video_length: int) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        config_path=str(cfg["config_path"]),
        model_name=str(cfg["model_name"]),
        video_length=int(video_length),
        sample_size=[int(v) for v in cfg["sample_size"]],
        fps=int(cfg["fps"]),
        guidance_scale=float(cfg["guidance_scale"]),
        num_inference_steps=int(cfg["num_inference_steps"]),
        negative_prompt=str(cfg["negative_prompt"]),
        gpu_memory_mode=str(cfg["gpu_memory_mode"]),
        sampler_name=str(cfg["sampler_name"]),
        shift=float(cfg["shift"]),
    )


def get_scheduler(name, scheduler_config):
    from diffusers import FlowMatchEulerDiscreteScheduler

    ensure_sys_path()
    from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
    from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
    from videox_fun.utils.utils import filter_kwargs

    scheduler_cls = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[name]
    return scheduler_cls(**filter_kwargs(scheduler_cls, scheduler_config))


def build_pipeline(args):
    import torch
    from omegaconf import OmegaConf

    ensure_sys_path()
    from videox_fun.models import (
        AutoTokenizer,
        AutoencoderKLWan,
        AutoencoderKLWan3_8,
        WanT5EncoderModel,
        Wan2_2Transformer3DModel,
    )
    from videox_fun.pipeline import Wan2_2FunInpaintPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = OmegaConf.load(args.config_path)
    boundary = config["transformer_additional_kwargs"].get("boundary", 0.900)

    transformer = Wan2_2Transformer3DModel.from_pretrained(
        os.path.join(
            args.model_name,
            config["transformer_additional_kwargs"].get("transformer_low_noise_model_subpath", "transformer"),
        ),
        transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=None,  # weights are loaded in float32 as in the paper runs (VideoX-Fun defaults to bf16)
    )
    if config["transformer_additional_kwargs"].get("transformer_combination_type", "single") == "moe":
        transformer_2 = Wan2_2Transformer3DModel.from_pretrained(
            os.path.join(
                args.model_name,
                config["transformer_additional_kwargs"].get("transformer_high_noise_model_subpath", "transformer"),
            ),
            transformer_additional_kwargs=OmegaConf.to_container(config["transformer_additional_kwargs"]),
            low_cpu_mem_usage=True,
            torch_dtype=None,
        )
    else:
        transformer_2 = None

    vae_cls = {
        "AutoencoderKLWan": AutoencoderKLWan,
        "AutoencoderKLWan3_8": AutoencoderKLWan3_8,
    }[config["vae_kwargs"].get("vae_type", "AutoencoderKLWan")]
    vae = vae_cls.from_pretrained(
        os.path.join(args.model_name, config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.model_name, config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=None,
    ).eval()

    scheduler_config = OmegaConf.to_container(config["scheduler_kwargs"])
    if args.sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
        scheduler_config["shift"] = 1
    scheduler = get_scheduler(args.sampler_name, scheduler_config)

    pipeline = Wan2_2FunInpaintPipeline(
        transformer=transformer,
        transformer_2=transformer_2,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )
    if args.gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    else:
        pipeline.to(device=device)

    return pipeline, device, boundary


def save_sample(sample, output_path, fps):
    ensure_sys_path()
    from videox_fun.utils.utils import save_videos_grid

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_videos_grid(sample, str(output_path), fps=fps)


def generate_segment(args, pipeline, vae, device, boundary, prompt, start_path, end_path, output_path, seed):
    import torch

    ensure_sys_path()
    from videox_fun.utils.utils import get_image_to_video_latent

    generator = torch.Generator(device=device).manual_seed(seed)
    ratio = vae.config.temporal_compression_ratio
    video_length = int((args.video_length - 1) // ratio * ratio) + 1

    input_video, input_video_mask, _ = get_image_to_video_latent(
        str(start_path),
        str(end_path) if end_path is not None else None,
        video_length=video_length,
        sample_size=args.sample_size,
    )

    with torch.no_grad():
        sample = pipeline(
            prompt,
            num_frames=video_length,
            negative_prompt=args.negative_prompt,
            height=args.sample_size[0],
            width=args.sample_size[1],
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            boundary=boundary,
            video=input_video,
            mask_video=input_video_mask,
            shift=args.shift,
        ).videos

    save_sample(sample, output_path, args.fps)
