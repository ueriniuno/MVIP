"""Training-time generator for Best-of-N scoring: diffusers Wan2.2-TI2V-5B."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import imageio.v2 as iio
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline
from PIL import Image

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
HEIGHT = 704
WIDTH = 1280
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.0
NEGATIVE_PROMPT = ""
FPS = 24


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_wan_pipeline(
    cache_dir: str | None = None,
    image_to_video: bool = False,
    force_cpu_offload: bool = False,
) -> Tuple[WanPipeline, str]:
    runtime_device = resolve_device()
    dtype = torch.bfloat16 if runtime_device == "cuda" else torch.float32
    pipeline_cls = WanImageToVideoPipeline if image_to_video else WanPipeline

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    )
    pipe = pipeline_cls.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    )

    if runtime_device == "cuda" and force_cpu_offload:
        pipe.enable_model_cpu_offload(device=runtime_device)
    else:
        pipe.to(runtime_device)

    return pipe, runtime_device


def normalize_num_frames(num_frames: int, max_num_frames: int) -> int:
    num_frames = max(5, min(int(num_frames), max_num_frames))
    remainder = (num_frames - 1) % 4
    if remainder:
        num_frames += 4 - remainder
    return min(num_frames, max_num_frames if (max_num_frames - 1) % 4 == 0 else max_num_frames - ((max_num_frames - 1) % 4))


def generate_video_frames(
    pipe: WanPipeline,
    device: str,
    prompt: str,
    seed: int,
    num_frames: int,
    init_image: Image.Image | None = None,
    last_image: Image.Image | None = None,
) -> List[Image.Image]:
    generator_device = "cuda" if device == "cuda" else "cpu"
    kwargs = {
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": num_frames,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "generator": torch.Generator(device=generator_device).manual_seed(seed),
    }
    if init_image is not None:
        kwargs["image"] = init_image
    if last_image is not None:
        kwargs["last_image"] = last_image
    return pipe(**kwargs).frames[0]


def save_video(frames: List[Image.Image], path: Path, fps: float) -> None:
    def to_uint8_array(frame: Image.Image) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.dtype == np.uint8:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).round().astype(np.uint8)
            return arr
        return np.clip(arr, 0, 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(path), fps=fps)
    try:
        for frame in frames:
            writer.append_data(to_uint8_array(frame))
    finally:
        writer.close()
