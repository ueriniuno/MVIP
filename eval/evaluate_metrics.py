#!/usr/bin/env python3
"""Evaluation metrics (CLIP-I, DINOv2, VideoMAE, DreamSim, RAFT, ViCLIP, SigLIP, BLIP2-ITM)
for every reference.mp4 / generated.mp4 pair under an output root of scripts/infer.py.

Usage:
    python eval/evaluate_metrics.py --output-root outputs/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]

TEXT_METRICS = ("viclip_vt", "siglip", "blip2_itm")
PAPER_METRICS = ("clip_i", "dinov2", "videomae", "dreamsim", "raft", "viclip_vv", "viclip_vt", "siglip", "blip2_itm")
METRIC_COLUMNS = {
    "clip_i": ["clip_i"],
    "dinov2": ["dinov2"],
    "videomae": ["videomae"],
    "dreamsim": ["dreamsim"],
    "raft": ["raft_motion_dist", "raft_mag_dist", "raft_dir_dist", "raft_ref_motion", "raft_gen_motion"],
    "viclip_vv": ["viclip_vv"],
    "viclip_vt": ["viclip_vt"],
    "siglip": ["siglip"],
    "blip2_itm": ["blip2_itm"],
}
SUMMARY_COLUMNS = [
    "clip_i", "dinov2", "videomae", "dreamsim", "raft_motion_dist", "raft_mag_dist", "raft_dir_dist",
    "raft_ref_motion", "raft_gen_motion", "dreamsim_std", "raft_num_pairs", "siglip", "blip2_itm",
    "viclip_vt", "viclip_vv",
]
LOWER_IS_BETTER = {"dreamsim", "raft_motion_dist", "raft_mag_dist", "raft_dir_dist"}


@dataclass(frozen=True)
class Pair:
    item_dir: Path
    ref_path: Path
    gen_path: Path
    caption_path: Path
    metrics_path: Path
    source: str
    video_id: str
    chunk_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", "--output_root", dest="output_root", type=Path, required=True)
    parser.add_argument("--metrics", default=",".join(PAPER_METRICS), help="Comma-separated subset of: " + ",".join(PAPER_METRICS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-image-frames", dest="num_image_frames", type=int, default=8)
    parser.add_argument("--num-video-frames", dest="num_video_frames", type=int, default=16)
    parser.add_argument("--num-viclip-frames", dest="num_viclip_frames", type=int, default=8)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=8)
    parser.add_argument("--model-cache-dir", dest="model_cache_dir", type=Path, default=REPO_ROOT / ".cache")
    parser.add_argument("--dreamsim-cache-dir", dest="dreamsim_cache_dir", type=Path, default=None)
    parser.add_argument("--skip-existing", dest="skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-unavailable-metrics", dest="skip_unavailable_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-pairs", dest="max_pairs", type=int, default=0)
    parser.add_argument("--summary-csv", dest="summary_csv", type=Path, default=None)
    parser.add_argument("--summary-json", dest="summary_json", type=Path, default=None)
    parser.add_argument("--clip_model_id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--dinov2_model_id", default="facebook/dinov2-base")
    parser.add_argument("--videomae_model_id", default="MCG-NJU/videomae-base")
    parser.add_argument("--raft_model", choices=["small", "large"], default="small")
    parser.add_argument("--raft_num_pairs", type=int, default=8)
    parser.add_argument("--raft_batch_size", type=int, default=2)
    parser.add_argument("--raft_long_side", type=int, default=384)
    parser.add_argument("--siglip_model_name", default="ViT-B-16-SigLIP-256")
    parser.add_argument("--siglip_pretrained", default="webli")
    parser.add_argument("--blip2_model_id", default="Salesforce/blip2-itm-vit-g")
    parser.add_argument("--viclip_model_id", default="OpenGVLab/ViCLIP-B-16-hf")
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.dreamsim_cache_dir is None:
        args.dreamsim_cache_dir = args.model_cache_dir / "dreamsim"
    return args


def parse_metrics(raw: str) -> list[str]:
    metrics = [m.strip() for m in raw.split(",") if m.strip()]
    if metrics == ["all"]:
        return list(PAPER_METRICS)
    unknown = [m for m in metrics if m not in METRIC_COLUMNS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; valid: {list(METRIC_COLUMNS)}")
    return metrics


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def discover_pairs(root: Path) -> list[Pair]:
    pairs: list[Pair] = []
    for gen_path in sorted(root.rglob("generated.mp4")):
        ref_path = gen_path.with_name("reference.mp4")
        if not ref_path.is_file():
            continue
        parts = gen_path.parent.relative_to(root).parts
        if "videos" in parts:
            idx = parts.index("videos")
            source = parts[idx + 1] if len(parts) > idx + 1 else ""
            video_id = parts[idx + 2] if len(parts) > idx + 2 else ""
            chunk_id = parts[idx + 3] if len(parts) > idx + 3 else gen_path.parent.name
        else:
            source = parts[0] if len(parts) > 2 else ""
            video_id = parts[-2] if len(parts) >= 2 else ""
            chunk_id = parts[-1] if parts else gen_path.parent.name
        pairs.append(Pair(gen_path.parent, ref_path, gen_path, gen_path.with_name("caption.txt"),
                          gen_path.with_name("all_metrics.json"), source, video_id, chunk_id))
    return pairs


def has_requested_metrics(payload: dict[str, Any], metrics: Iterable[str]) -> bool:
    return all(column in payload for metric in metrics for column in METRIC_COLUMNS[metric])


# ---------------------------------------------------------------------------
# Frame loading and the frame-level / video-level reconstruction metrics
# ---------------------------------------------------------------------------
def load_video_rgb(video_path: Path) -> list[np.ndarray]:
    reader = iio.get_reader(str(video_path))
    try:
        frames = [np.asarray(frame)[..., :3].astype(np.uint8) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames in {video_path}")
    return frames


def sample_indices(total: int, count: int, exact: bool = True) -> list[int]:
    if total <= 0:
        return []
    if count <= 0:
        return []
    if exact:
        return [int(idx) for idx in np.linspace(0, total - 1, count, dtype=int)]
    count = min(count, total)
    return [int(idx) for idx in np.linspace(0, total - 1, count, dtype=int)]


def frames_to_pil(frames: list[np.ndarray], indices: list[int]) -> list[Image.Image]:
    return [Image.fromarray(frames[idx]).convert("RGB") for idx in indices]


def pair_frame_sets(
    ref_frames: list[np.ndarray],
    gen_frames: list[np.ndarray],
    count: int,
    exact: bool = True,
) -> tuple[list[Image.Image], list[Image.Image]]:
    ref_indices = sample_indices(len(ref_frames), count, exact=exact)
    gen_indices = sample_indices(len(gen_frames), count, exact=exact)
    n = min(len(ref_indices), len(gen_indices))
    return frames_to_pil(ref_frames, ref_indices[:n]), frames_to_pil(gen_frames, gen_indices[:n])


def cosine_mean(ref_feats: torch.Tensor, gen_feats: torch.Tensor) -> float:
    def output_to_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        image_embeds = getattr(value, "image_embeds", None)
        if image_embeds is not None:
            return image_embeds
        pooler_output = getattr(value, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        return value.last_hidden_state[:, 0, :]

    ref_feats = output_to_tensor(ref_feats)
    gen_feats = output_to_tensor(gen_feats)
    ref_feats = ref_feats / ref_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    gen_feats = gen_feats / gen_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return float((ref_feats * gen_feats).sum(dim=-1).mean().item())


def load_clip(args: argparse.Namespace):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(
        args.clip_model_id,
        cache_dir=str(args.model_cache_dir),
    ).to(args.device)
    processor = CLIPProcessor.from_pretrained(
        args.clip_model_id,
        cache_dir=str(args.model_cache_dir),
    )
    model.eval()
    return model, processor


@torch.no_grad()
def score_clip_i(
    ref_frames: list[Image.Image],
    gen_frames: list[Image.Image],
    model: Any,
    processor: Any,
    device: str,
) -> float:
    def embed(frames: list[Image.Image]) -> torch.Tensor:
        inputs = processor(images=frames, return_tensors="pt", padding=True).to(device)
        return model.get_image_features(**inputs)

    return round(cosine_mean(embed(ref_frames), embed(gen_frames)), 6)


def load_dinov2(args: argparse.Namespace):
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(
        args.dinov2_model_id,
        cache_dir=str(args.model_cache_dir),
    ).to(args.device)
    processor = AutoImageProcessor.from_pretrained(
        args.dinov2_model_id,
        cache_dir=str(args.model_cache_dir),
    )
    model.eval()
    return model, processor


@torch.no_grad()
def score_dinov2(
    ref_frames: list[Image.Image],
    gen_frames: list[Image.Image],
    model: Any,
    processor: Any,
    device: str,
) -> float:
    def embed(frames: list[Image.Image]) -> torch.Tensor:
        inputs = processor(images=frames, return_tensors="pt").to(device)
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

    return round(cosine_mean(embed(ref_frames), embed(gen_frames)), 6)


def load_videomae(args: argparse.Namespace):
    from transformers import AutoImageProcessor, VideoMAEModel

    processor = AutoImageProcessor.from_pretrained(
        args.videomae_model_id,
        cache_dir=str(args.model_cache_dir),
    )
    model = VideoMAEModel.from_pretrained(
        args.videomae_model_id,
        cache_dir=str(args.model_cache_dir),
    ).to(args.device)
    model.eval()
    return model, processor


@torch.no_grad()
def score_videomae(
    ref_frames: list[Image.Image],
    gen_frames: list[Image.Image],
    model: Any,
    processor: Any,
    device: str,
) -> float:
    def embed(frames: list[Image.Image]) -> torch.Tensor:
        inputs = processor(frames, return_tensors="pt").to(device)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    return round(cosine_mean(embed(ref_frames), embed(gen_frames)), 6)


def load_dreamsim(args: argparse.Namespace):
    from dreamsim import dreamsim

    args.dreamsim_cache_dir.mkdir(parents=True, exist_ok=True)
    model, preprocess = dreamsim(
        pretrained=True,
        device=args.device,
        cache_dir=str(args.dreamsim_cache_dir),
    )
    model.eval()
    return model, preprocess


def dreamsim_preprocess_batch(
    frames: list[Image.Image],
    preprocess: Any,
    device: str,
) -> torch.Tensor:
    tensors = []
    for frame in frames:
        tensor = preprocess(frame)
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        tensors.append(tensor)
    return torch.stack(tensors, dim=0).to(device)


@torch.no_grad()
def score_dreamsim(
    ref_frames: list[Image.Image],
    gen_frames: list[Image.Image],
    model: Any,
    preprocess: Any,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    distances: list[float] = []
    for start in range(0, len(ref_frames), batch_size):
        ref_batch = dreamsim_preprocess_batch(ref_frames[start:start + batch_size], preprocess, device)
        gen_batch = dreamsim_preprocess_batch(gen_frames[start:start + batch_size], preprocess, device)
        out = model(ref_batch, gen_batch).detach().float().view(-1).cpu().numpy().tolist()
        distances.extend(float(value) for value in out)
    return {
        "dreamsim": round(float(np.mean(distances)), 6),
        "dreamsim_std": round(float(np.std(distances)), 6),
    }


def load_raft(args: argparse.Namespace):
    if args.raft_model == "large":
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=True).to(args.device)
    else:
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=True).to(args.device)
    model.eval()
    return model, weights.transforms()


def round_to_multiple(value: int, multiple: int) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


def raft_target_size(frame: Image.Image, long_side: int) -> tuple[int, int]:
    width, height = frame.size
    if long_side <= 0:
        return round_to_multiple(height, 8), round_to_multiple(width, 8)
    scale = long_side / max(width, height)
    target_h = round_to_multiple(max(8, int(round(height * scale))), 8)
    target_w = round_to_multiple(max(8, int(round(width * scale))), 8)
    return target_h, target_w


def resize_for_raft(frame: Image.Image, target_h: int, target_w: int) -> Image.Image:
    return frame.resize((target_w, target_h), Image.BICUBIC)


def pair_start_indices(total: int, count: int) -> list[int]:
    if total < 2 or count <= 0:
        return []
    count = min(count, total - 1)
    return [int(idx) for idx in np.linspace(0, total - 2, count, dtype=int)]


def pil_batch_to_uint8(frames: list[Image.Image]) -> torch.Tensor:
    tensors = []
    for frame in frames:
        arr = np.asarray(frame.convert("RGB"), dtype=np.uint8)
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def raft_forward(
    model: Any,
    transforms: Any,
    first_frames: list[Image.Image],
    second_frames: list[Image.Image],
    device: str,
    batch_size: int,
) -> torch.Tensor:
    flows = []
    for start in range(0, len(first_frames), batch_size):
        img1 = pil_batch_to_uint8(first_frames[start:start + batch_size])
        img2 = pil_batch_to_uint8(second_frames[start:start + batch_size])
        img1, img2 = transforms(img1, img2)
        flow = model(img1.to(device), img2.to(device))[-1]
        flows.append(flow.detach().float().cpu())
    return torch.cat(flows, dim=0)


def build_raft_pairs(
    frames: list[np.ndarray],
    starts: list[int],
    target_h: int,
    target_w: int,
) -> tuple[list[Image.Image], list[Image.Image]]:
    first, second = [], []
    for idx in starts:
        first.append(resize_for_raft(Image.fromarray(frames[idx]).convert("RGB"), target_h, target_w))
        second.append(resize_for_raft(Image.fromarray(frames[idx + 1]).convert("RGB"), target_h, target_w))
    return first, second


@torch.no_grad()
def score_raft(
    ref_np: list[np.ndarray],
    gen_np: list[np.ndarray],
    model: Any,
    transforms: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    ref_starts = pair_start_indices(len(ref_np), args.raft_num_pairs)
    gen_starts = pair_start_indices(len(gen_np), args.raft_num_pairs)
    n = min(len(ref_starts), len(gen_starts))
    if n == 0:
        return {
            "raft_motion_dist": None,
            "raft_mag_dist": None,
            "raft_dir_dist": None,
            "raft_ref_motion": None,
            "raft_gen_motion": None,
            "raft_num_pairs": 0,
        }

    ref_frame = Image.fromarray(ref_np[ref_starts[0]]).convert("RGB")
    target_h, target_w = raft_target_size(ref_frame, args.raft_long_side)
    ref_1, ref_2 = build_raft_pairs(ref_np, ref_starts[:n], target_h, target_w)
    gen_1, gen_2 = build_raft_pairs(gen_np, gen_starts[:n], target_h, target_w)

    ref_flow = raft_forward(model, transforms, ref_1, ref_2, args.device, args.raft_batch_size)
    gen_flow = raft_forward(model, transforms, gen_1, gen_2, args.device, args.raft_batch_size)

    ref_mag = torch.linalg.vector_norm(ref_flow, dim=1)
    gen_mag = torch.linalg.vector_norm(gen_flow, dim=1)
    eps = 1e-6

    mag_dist = torch.mean(torch.abs(ref_mag - gen_mag)) / (
        torch.mean(ref_mag) + torch.mean(gen_mag) + eps
    )
    dot = torch.sum(ref_flow * gen_flow, dim=1)
    denom = (ref_mag * gen_mag).clamp_min(eps)
    cos = torch.clamp(dot / denom, -1.0, 1.0)
    active = (ref_mag + gen_mag) > 0.05
    if bool(active.any()):
        dir_dist = torch.mean((1.0 - cos[active]) * 0.5)
    else:
        dir_dist = torch.tensor(0.0)
    motion_dist = 0.5 * mag_dist + 0.5 * dir_dist

    return {
        "raft_motion_dist": round(float(motion_dist.item()), 6),
        "raft_mag_dist": round(float(mag_dist.item()), 6),
        "raft_dir_dist": round(float(dir_dist.item()), 6),
        "raft_ref_motion": round(float(torch.mean(ref_mag).item()), 6),
        "raft_gen_motion": round(float(torch.mean(gen_mag).item()), 6),
        "raft_num_pairs": n,
        "raft_size": [target_h, target_w],
    }


# ---------------------------------------------------------------------------
# ViCLIP, SigLIP and BLIP2-ITM
# ---------------------------------------------------------------------------
def load_open_clip(args: argparse.Namespace, model_name: str, pretrained: str):
    import open_clip

    pretrained_arg = pretrained or None
    if model_name.startswith("hf-hub:"):
        model, preprocess = open_clip.create_model_from_pretrained(
            model_name,
            pretrained=pretrained_arg,
            device=args.device,
            cache_dir=str(args.model_cache_dir),
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_arg,
            device=args.device,
            cache_dir=str(args.model_cache_dir),
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def load_viclip(args: argparse.Namespace):
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoModel
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(
        args.viclip_model_id,
        cache_dir=str(args.model_cache_dir),
        trust_remote_code=args.trust_remote_code,
    )
    try:
        config.tokenizer_path = hf_hub_download(
            args.viclip_model_id,
            "bpe_simple_vocab_16e6.txt.gz",
            cache_dir=str(args.model_cache_dir),
        )
    except Exception:
        pass
    model_class = get_class_from_dynamic_module(
        config.auto_map["AutoModel"],
        args.viclip_model_id,
        cache_dir=str(args.model_cache_dir),
        trust_remote_code=args.trust_remote_code,
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        # Required by newer transformers; ViCLIP's remote code does not define it.
        model_class.all_tied_weights_keys = {}
    model = AutoModel.from_pretrained(
        args.viclip_model_id,
        config=config,
        cache_dir=str(args.model_cache_dir),
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    for block in getattr(getattr(model, "text_encoder", None), "transformer", object()).resblocks:
        mask = getattr(block, "attn_mask", None)
        if getattr(mask, "is_meta", False):
            context_length = getattr(model.text_encoder, "context_length", mask.shape[0])
            block.attn_mask = torch.empty(context_length, context_length).fill_(float("-inf")).triu_(1)
    model.eval()
    return model


def load_blip2_itm(args: argparse.Namespace):
    from transformers import Blip2ForImageTextRetrieval, Blip2Processor

    dtype = torch.float16 if str(args.device).startswith("cuda") else torch.float32
    processor = Blip2Processor.from_pretrained(args.blip2_model_id, cache_dir=str(args.model_cache_dir))
    model = Blip2ForImageTextRetrieval.from_pretrained(
        args.blip2_model_id,
        cache_dir=str(args.model_cache_dir),
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()
    return model, processor


def pil_from_np_frames(frames: list[np.ndarray], count: int) -> list[Image.Image]:
    indices = sample_indices(len(frames), count, exact=True)
    return [Image.fromarray(frames[idx]).convert("RGB") for idx in indices]


def l2_normalize(features: torch.Tensor) -> torch.Tensor:
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)


@torch.no_grad()
def score_open_clip_text_video(
    caption: str,
    frames: list[Image.Image],
    model: Any,
    preprocess: Any,
    tokenizer: Any,
    device: str,
    batch_size: int,
) -> float:
    text_tokens = tokenizer([caption]).to(device)
    text_features = l2_normalize(model.encode_text(text_tokens).float())
    scores: list[float] = []
    for start in range(0, len(frames), batch_size):
        batch = torch.stack([preprocess(frame) for frame in frames[start:start + batch_size]], dim=0).to(device)
        image_features = l2_normalize(model.encode_image(batch).float())
        sims = image_features @ text_features.T
        scores.extend(float(value) for value in sims.view(-1).detach().cpu().tolist())
    return round(float(np.mean(scores)), 6)


def clip_video_transform(size: int = 224):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def video_tensor_for_clip(frames_np: list[np.ndarray], count: int, device: str, size: int = 224) -> torch.Tensor:
    transform = clip_video_transform(size)
    frames = pil_from_np_frames(frames_np, count)
    tensor = torch.stack([transform(frame) for frame in frames], dim=0)
    return tensor.unsqueeze(0).to(device)


def encode_text_generic(model: Any, caption: str) -> torch.Tensor:
    if hasattr(model, "get_text_features"):
        try:
            return model.get_text_features(caption, None).float()
        except TypeError:
            return model.get_text_features(caption).float()
    if hasattr(model, "encode_text"):
        return model.encode_text(caption).float()
    raise AttributeError("model has no supported text encoder method")


def encode_video_generic(model: Any, video_tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "get_vid_features"):
        return model.get_vid_features(video_tensor).float()
    if hasattr(model, "get_video_features"):
        return model.get_video_features(video_tensor).float()
    if hasattr(model, "encode_video"):
        return model.encode_video(video_tensor).float()
    if hasattr(model, "encode_vision"):
        try:
            return model.encode_vision(video_tensor, test=True).float()
        except TypeError:
            return model.encode_vision(video_tensor).float()
    raise AttributeError("model has no supported video encoder method")


def pool_feature(features: torch.Tensor) -> torch.Tensor:
    if features.ndim > 2:
        reduce_dims = tuple(range(1, features.ndim - 1))
        features = features.mean(dim=reduce_dims)
    return features.view(features.shape[0], -1)


@torch.no_grad()
def score_video_text_model(
    model: Any,
    caption: str,
    ref_np: list[np.ndarray],
    device: str,
    num_frames: int,
) -> float:
    video_tensor = video_tensor_for_clip(ref_np, num_frames, device)
    video_features = l2_normalize(pool_feature(encode_video_generic(model, video_tensor)))
    text_features = l2_normalize(pool_feature(encode_text_generic(model, caption)))
    return round(float((video_features @ text_features.T).item()), 6)


@torch.no_grad()
def score_video_video_model(
    model: Any,
    ref_np: list[np.ndarray],
    gen_np: list[np.ndarray],
    device: str,
    num_frames: int,
) -> float:
    ref_tensor = video_tensor_for_clip(ref_np, num_frames, device)
    gen_tensor = video_tensor_for_clip(gen_np, num_frames, device)
    ref_features = l2_normalize(pool_feature(encode_video_generic(model, ref_tensor)))
    gen_features = l2_normalize(pool_feature(encode_video_generic(model, gen_tensor)))
    return round(float((ref_features @ gen_features.T).item()), 6)


@torch.no_grad()
def score_blip2_itm(
    caption: str,
    frames: list[Image.Image],
    model: Any,
    processor: Any,
    device: str,
    batch_size: int,
) -> float:
    scores: list[float] = []
    for start in range(0, len(frames), batch_size):
        batch_frames = frames[start:start + batch_size]
        texts = [caption] * len(batch_frames)
        inputs = processor(images=batch_frames, text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        outputs = model(**inputs, use_image_text_matching_head=True)
        logits = outputs.logits_per_image
        if logits is None:
            logits = outputs.logits_per_text
        logits = logits.float()
        if logits.ndim >= 2 and logits.shape[-1] == 2:
            values = F.softmax(logits, dim=-1)[..., 1]
        elif logits.ndim == 2 and logits.shape[0] == logits.shape[1]:
            values = logits.diag()
        else:
            values = logits.view(-1)
        scores.extend(float(value) for value in values.detach().cpu().tolist())
    return round(float(np.mean(scores)), 6)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_models(args: argparse.Namespace, metrics: list[str]) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    loaders = {
        "clip_i": lambda: load_clip(args),
        "dinov2": lambda: load_dinov2(args),
        "videomae": lambda: load_videomae(args),
        "dreamsim": lambda: load_dreamsim(args),
        "raft": lambda: load_raft(args),
        "siglip": lambda: load_open_clip(args, args.siglip_model_name, args.siglip_pretrained),
        "blip2_itm": lambda: load_blip2_itm(args),
        "viclip_vt": lambda: load_viclip(args),
        "viclip_vv": lambda: load_viclip(args),
    }
    aliases = {"viclip_vv": "viclip_vt", "viclip_vt": "viclip_vv"}
    models: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for metric in metrics:
        alias = aliases.get(metric)
        if alias in models:
            models[metric] = models[alias]
            continue
        print(f"[*] Loading {metric}", flush=True)
        try:
            models[metric] = loaders[metric]()
        except Exception as exc:
            if not args.skip_unavailable_metrics:
                raise
            errors[metric] = str(exc)
            print(f"[WARN] Skipping unavailable metric {metric}: {exc}", flush=True)
    loaded = [metric for metric in metrics if metric in models]
    return models, loaded, errors


def evaluate_pair(pair: Pair, models: dict[str, Any], requested: list[str], loaded: list[str], args: argparse.Namespace) -> dict[str, Any]:
    ref_np = load_video_rgb(pair.ref_path)
    gen_np = load_video_rgb(pair.gen_path)
    ref_img, gen_img = pair_frame_sets(ref_np, gen_np, args.num_image_frames, exact=True)
    ref_vid, gen_vid = pair_frame_sets(ref_np, gen_np, args.num_video_frames, exact=True)
    caption = pair.caption_path.read_text(encoding="utf-8").strip() if pair.caption_path.is_file() else ""

    result: dict[str, Any] = {
        "source": pair.source,
        "video_id": pair.video_id,
        "chunk_id": pair.chunk_id,
        "ref_path": str(pair.ref_path),
        "gen_path": str(pair.gen_path),
        "num_image_frames": len(ref_img),
        "num_video_frames": len(ref_vid),
        "metrics_requested": list(requested),
        "metrics_loaded": list(loaded),
        "metric_directions": {column: ("lower" if column in LOWER_IS_BETTER else "higher") for column in SUMMARY_COLUMNS},
    }
    if "clip_i" in models:
        model, processor = models["clip_i"]
        result["clip_i"] = score_clip_i(ref_img, gen_img, model, processor, args.device)
    if "dinov2" in models:
        model, processor = models["dinov2"]
        result["dinov2"] = score_dinov2(ref_img, gen_img, model, processor, args.device)
    if "videomae" in models:
        model, processor = models["videomae"]
        result["videomae"] = score_videomae(ref_vid, gen_vid, model, processor, args.device)
    if "dreamsim" in models:
        model, preprocess = models["dreamsim"]
        result.update(score_dreamsim(ref_img, gen_img, model, preprocess, args.device, args.batch_size))
    if "raft" in models:
        model, transforms = models["raft"]
        result.update(score_raft(ref_np, gen_np, model, transforms, args))
    for metric in ("viclip_vv", "viclip_vt", "siglip", "blip2_itm"):
        if metric not in models:
            continue
        if metric in TEXT_METRICS and not caption:
            result[f"{metric}_error"] = "caption.txt missing or empty"
            continue
        try:
            if metric == "viclip_vv":
                result["viclip_vv"] = score_video_video_model(models[metric], ref_np, gen_np, args.device, args.num_viclip_frames)
            elif metric == "viclip_vt":
                result["viclip_vt"] = score_video_text_model(models[metric], caption, ref_np, args.device, args.num_viclip_frames)
            elif metric == "siglip":
                model, preprocess, tokenizer = models[metric]
                result["siglip"] = score_open_clip_text_video(caption, ref_img, model, preprocess, tokenizer, args.device, args.batch_size)
            elif metric == "blip2_itm":
                model, processor = models[metric]
                result["blip2_itm"] = score_blip2_itm(caption, ref_img, model, processor, args.device, args.batch_size)
        except Exception as exc:
            print(f"[WARN] {metric} failed for {pair.item_dir}: {exc}", flush=True)
            result[f"{metric}_error"] = str(exc)
    return result


def summary_row(payload: dict[str, Any]) -> dict[str, Any]:
    row = {key: payload.get(key, "") for key in ("source", "video_id", "chunk_id")}
    for column in SUMMARY_COLUMNS:
        row[column] = payload.get(column, "")
    row["ref_path"] = payload.get("ref_path", "")
    row["gen_path"] = payload.get("gen_path", "")
    return row


def mean_of(rows: list[dict[str, Any]], column: str) -> Optional[float]:
    values = [float(r[column]) for r in rows if r.get(column) not in ("", None) and math.isfinite(float(r[column]))]
    return round(float(np.mean(values)), 6) if values else None


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    pairs = discover_pairs(root)
    if not pairs:
        raise SystemExit(f"No reference.mp4/generated.mp4 pairs found under {root}")
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    requested = parse_metrics(args.metrics)
    print(f"[*] {len(pairs)} pairs under {root}; metrics: {', '.join(requested)}", flush=True)

    pending = [p for p in pairs if not (args.skip_existing and p.metrics_path.is_file() and has_requested_metrics(read_json(p.metrics_path), requested))]
    models, loaded, errors = ({}, [], {})
    if pending:
        models, loaded, errors = load_models(args, requested)

    rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        existing = read_json(pair.metrics_path) if pair.metrics_path.is_file() else {}
        if pair in pending:
            payload = {**existing, **evaluate_pair(pair, models, requested, loaded, args)}
            write_json(pair.metrics_path, payload)
            shown = " ".join(f"{c}={payload[c]:.4f}" for c in ("clip_i", "dinov2", "videomae", "dreamsim", "viclip_vv") if isinstance(payload.get(c), float))
            print(f"[{idx}/{len(pairs)}] {pair.source}/{pair.video_id}/{pair.chunk_id} {shown}", flush=True)
        else:
            payload = existing
            print(f"[{idx}/{len(pairs)}] {pair.source}/{pair.video_id}/{pair.chunk_id} (existing)", flush=True)
        rows.append(summary_row(payload))

    summary_csv = args.summary_csv or root / "metrics_summary.csv"
    summary_json = args.summary_json or root / "metrics_summary.json"
    with open(summary_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    write_json(summary_json, {
        "num_pairs": len(rows),
        "metrics_requested": requested,
        "metrics_loaded": loaded,
        "load_errors": errors,
        "mean": {c: mean_of(rows, c) for c in SUMMARY_COLUMNS},
        "mean_by_source": {s: {c: mean_of(r, c) for c in SUMMARY_COLUMNS} for s, r in sorted(by_source.items())},
    })
    print(f"[*] wrote {summary_csv} and {summary_json}", flush=True)


if __name__ == "__main__":
    main()
