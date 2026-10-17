from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

try:
    import cv2
except Exception:  # optional frame-decoding fallback
    cv2 = None


# ---- sample catalog ----
@dataclass(frozen=True)
class CaptionSample:
    sample_id: str
    source: str
    original_video_id: str
    segment_id: str | None
    file_path: Path
    duration: float
    split: str
    start_sec: float | None = None
    end_sec: float | None = None

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["file_path"] = str(self.file_path)
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "CaptionSample":
        return cls(
            sample_id=str(payload["sample_id"]),
            source=str(payload["source"]),
            original_video_id=str(payload["original_video_id"]),
            segment_id=payload.get("segment_id"),
            file_path=Path(payload["file_path"]),
            duration=float(payload.get("duration", 0.0)),
            split=str(payload.get("split", "train")),
            start_sec=None if payload.get("start_sec") is None else float(payload["start_sec"]),
            end_sec=None if payload.get("end_sec") is None else float(payload["end_sec"]),
        )


# ---- VLM loading / generated-frame sampling ----
def load_vlm_model(
    model_id: str, cache_dir: str | None = None
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", cache_dir=cache_dir
    )
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    return model, processor

def sample_generated_frames(frames: list[Any], max_frames: int) -> list[Image.Image]:
    if max_frames <= 0 or len(frames) == 0:
        return []
    if max_frames == 1:
        selected = [frames[len(frames) // 2]]
    elif len(frames) <= max_frames:
        selected = frames
    else:
        step = (len(frames) - 1) / float(max_frames - 1)
        indices = sorted({int(round(i * step)) for i in range(max_frames)})
        selected = [frames[idx] for idx in indices[:max_frames]]
    result: list[Image.Image] = []
    for frame in selected:
        if isinstance(frame, Image.Image):
            result.append(frame.convert("RGB"))
        else:
            arr = frame
            if hasattr(arr, "dtype") and str(arr.dtype).startswith("float"):
                import numpy as np
                arr = (np.clip(arr, 0.0, 1.0) * 255.0).round().astype("uint8")
            result.append(Image.fromarray(arr).convert("RGB"))
    while len(result) < max_frames:
        result.append(result[-1].copy())
    return result

# ---- tasks, schedules, frame windows, run state ----
@dataclass(frozen=True)
class ClipCaptionSample:
    sample_id: str
    source: str
    original_video_id: str
    segment_id: str | None
    file_path: Path
    duration: float
    split: str
    base_sample_id: str
    start_frame_idx: int
    end_frame_idx: int
    start_sec: float | None = None
    end_sec: float | None = None

    def to_json(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "original_video_id": self.original_video_id,
            "segment_id": self.segment_id,
            "file_path": str(self.file_path),
            "duration": self.duration,
            "split": self.split,
            "base_sample_id": self.base_sample_id,
            "start_frame_idx": self.start_frame_idx,
            "end_frame_idx": self.end_frame_idx,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "ClipCaptionSample":
        return cls(
            sample_id=str(payload["sample_id"]),
            source=str(payload["source"]),
            original_video_id=str(payload["original_video_id"]),
            segment_id=payload.get("segment_id"),
            file_path=Path(payload["file_path"]),
            duration=float(payload.get("duration", 0.0)),
            split=str(payload.get("split", "train")),
            base_sample_id=str(payload.get("base_sample_id", payload["sample_id"])),
            start_frame_idx=int(payload["start_frame_idx"]),
            end_frame_idx=int(payload["end_frame_idx"]),
            start_sec=None if payload.get("start_sec") is None else float(payload["start_sec"]),
            end_sec=None if payload.get("end_sec") is None else float(payload["end_sec"]),
        )

@dataclass
class TaskBatch:
    task_name: str
    task_mode: str
    task_samples: list[CaptionSample]
    support_samples: list[ClipCaptionSample]
    query_samples: list[ClipCaptionSample]
    source_distribution: dict[str, int]

    def to_json(self) -> dict:
        return {
            "task_name": self.task_name,
            "task_mode": self.task_mode,
            "task_samples": [sample.to_json() for sample in self.task_samples],
            "support_samples": [sample.to_json() for sample in self.support_samples],
            "query_samples": [sample.to_json() for sample in self.query_samples],
            "source_distribution": self.source_distribution,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "TaskBatch":
        return cls(
            task_name=str(payload["task_name"]),
            task_mode=str(payload["task_mode"]),
            task_samples=[CaptionSample.from_json(item) for item in payload.get("task_samples", [])],
            support_samples=[ClipCaptionSample.from_json(item) for item in payload.get("support_samples", [])],
            query_samples=[ClipCaptionSample.from_json(item) for item in payload.get("query_samples", [])],
            source_distribution={str(key): int(value) for key, value in payload.get("source_distribution", {}).items()},
        )

def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

def state_root_for_run(out_root: Path) -> Path:
    return out_root / "state"

def schedule_path_for_outer_step(out_root: Path, outer_step: int) -> Path:
    return state_root_for_run(out_root) / f"outer_step_{outer_step:03d}_schedule.json"

def run_state_path(out_root: Path) -> Path:
    return state_root_for_run(out_root) / "run_state.json"

def save_task_schedule(path: Path, task_batches: list[TaskBatch]) -> None:
    write_json(path, {"task_batches": [batch.to_json() for batch in task_batches]})

def load_task_schedule(path: Path) -> list[TaskBatch]:
    payload = read_json(path, default={"task_batches": []})
    return [TaskBatch.from_json(item) for item in payload.get("task_batches", [])]

def order_history_entries(history_entries: dict[int, dict]) -> list[dict]:
    return [history_entries[step] for step in sorted(history_entries)]


def resolve_sample_frame_window(sample: CaptionSample | ClipCaptionSample) -> tuple[int, int, float]:
    reader = iio.get_reader(str(sample.file_path))
    try:
        total_frames = reader.count_frames()
        if total_frames <= 0:
            raise ValueError(f"no frames detected in {sample.file_path}")
        meta = reader.get_meta_data()
        fps = float(meta.get("fps") or 0.0)
        duration_sec = float(meta.get("duration") or 0.0)

        if isinstance(sample, ClipCaptionSample):
            start_idx = max(0, min(sample.start_frame_idx, total_frames - 1))
            end_idx = max(start_idx + 1, min(sample.end_frame_idx, total_frames))
            return start_idx, end_idx, fps

        start_idx = 0
        end_idx = total_frames
        if sample.start_sec is not None and sample.end_sec is not None:
            if fps > 0:
                start_idx = max(0, int(math.floor(sample.start_sec * fps)))
                end_idx = min(total_frames, int(math.ceil(sample.end_sec * fps)))
            elif duration_sec > 0:
                start_ratio = min(max(sample.start_sec / duration_sec, 0.0), 1.0)
                end_ratio = min(max(sample.end_sec / duration_sec, 0.0), 1.0)
                start_idx = max(0, int(math.floor(total_frames * start_ratio)))
                end_idx = min(total_frames, int(math.ceil(total_frames * end_ratio)))
            if end_idx <= start_idx:
                end_idx = min(total_frames, start_idx + 1)
        return start_idx, end_idx, fps
    finally:
        reader.close()

def resolve_gen_num_frames(sample: CaptionSample | ClipCaptionSample, args: argparse.Namespace) -> int:
    gen_num_frames = args.num_gen_frames
    if not getattr(args, "match_ref_num_frames", False):
        return gen_num_frames
    start_idx, end_idx, fps = resolve_sample_frame_window(sample)
    ref_frames = max(1, end_idx - start_idx)
    max_gen_seconds = float(getattr(args, "max_gen_seconds", 0.0) or 0.0)
    if max_gen_seconds > 0 and fps > 0:
        cap_frames = max(1, int(round(max_gen_seconds * fps)))
        ref_frames = min(ref_frames, cap_frames)
    max_num_gen_frames = int(getattr(args, "max_num_gen_frames", 0) or ref_frames)
    ref_frames = min(ref_frames, max_num_gen_frames)
    normalized = max(5, int(ref_frames))
    remainder = (normalized - 1) % 4
    if remainder:
        normalized += 4 - remainder
    if normalized > max_num_gen_frames:
        max_aligned = max_num_gen_frames - ((max_num_gen_frames - 1) % 4)
        normalized = max(5, max_aligned)
    return normalized

def split_frame_window(
    *,
    start_idx: int,
    end_idx: int,
    total_clips: int,
    min_frames_per_clip: int,
) -> list[tuple[int, int]]:
    total_frames = end_idx - start_idx
    min_total_frames = max(1, total_clips * min_frames_per_clip)
    if total_frames < min_total_frames:
        raise ValueError(
            f"not enough frames to create {total_clips} clips: have {total_frames}, need {min_total_frames}"
        )

    clip_ranges: list[tuple[int, int]] = []
    for clip_idx in range(total_clips):
        clip_start = start_idx + int(round(clip_idx * total_frames / float(total_clips)))
        clip_end = start_idx + int(round((clip_idx + 1) * total_frames / float(total_clips)))
        if clip_end <= clip_start:
            clip_end = min(end_idx, clip_start + 1)
        clip_ranges.append((clip_start, clip_end))
    return clip_ranges

def build_clip_sample(
    *,
    base_sample: CaptionSample,
    clip_index: int,
    clip_start_idx: int,
    clip_end_idx: int,
    fps: float,
) -> ClipCaptionSample:
    clip_suffix = f"clip_{clip_index:02d}"
    clip_segment_id = clip_suffix if base_sample.segment_id is None else f"{base_sample.segment_id}_{clip_suffix}"
    start_sec = (clip_start_idx / fps) if fps > 0 else None
    end_sec = (clip_end_idx / fps) if fps > 0 else None
    duration = max(0.0, ((clip_end_idx - clip_start_idx) / fps) if fps > 0 else 0.0)
    return ClipCaptionSample(
        sample_id=f"{base_sample.sample_id}:{clip_suffix}",
        source=base_sample.source,
        original_video_id=base_sample.original_video_id,
        segment_id=clip_segment_id,
        file_path=base_sample.file_path,
        duration=duration,
        split=base_sample.split,
        base_sample_id=base_sample.sample_id,
        start_frame_idx=clip_start_idx,
        end_frame_idx=clip_end_idx,
        start_sec=start_sec,
        end_sec=end_sec,
    )

def load_uniform_sample_frames(sample: CaptionSample | ClipCaptionSample, num_frames: int) -> list[Image.Image]:
    start_idx, end_idx, _ = resolve_sample_frame_window(sample)
    indices = uniform_indices(start_idx, end_idx, num_frames)
    return load_frames_by_indices(sample.file_path, indices)

def load_frames_by_indices(video_path: Path, indices: list[int]) -> list[Image.Image]:
    if cv2 is not None:
        frames = load_frames_by_indices_cv2(video_path, indices)
        if frames:
            return frames
    reader = iio.get_reader(str(video_path))
    try:
        frames: list[Image.Image] = []
        for idx in indices:
            arr = reader.get_data(int(idx))
            frames.append(Image.fromarray(arr).convert("RGB"))
        return frames
    finally:
        reader.close()

def load_frames_by_indices_cv2(video_path: Path, indices: list[int]) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return []
        frames: list[Image.Image] = []
        last_frame: Image.Image | None = None
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_frame is None:
                    return []
                frames.append(last_frame.copy())
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).convert("RGB")
            frames.append(image)
            last_frame = image
        return frames
    finally:
        cap.release()

def uniform_indices(start_idx: int, end_idx: int, num_frames: int) -> list[int]:
    count = max(1, end_idx - start_idx)
    if count == 1:
        return [start_idx] * num_frames
    if count >= num_frames:
        step = (count - 1) / float(num_frames - 1)
        return [start_idx + int(round(i * step)) for i in range(num_frames)]
    indices = [start_idx + i for i in range(count)]
    while len(indices) < num_frames:
        indices.append(indices[-1])
    return indices[:num_frames]

# ---- task construction ----

def build_support_only_task_from_sample(
    *,
    sample: CaptionSample,
    task_mode: str,
    support_size: int,
    min_frames_per_clip: int,
) -> TaskBatch | None:
    if task_mode != "same_sample_clip":
        raise ValueError(f"Unsupported task_mode: {task_mode}")
    if support_size <= 0:
        raise ValueError("support_size must be positive for support-only Reptile training")

    try:
        start_idx, end_idx, fps = resolve_sample_frame_window(sample)
        total_frames = max(1, end_idx - start_idx)
        if total_frames < min_frames_per_clip:
            return None

        if support_size == 1:
            support_ranges = [(start_idx, end_idx)]
        else:
            support_ranges = split_frame_window(
                start_idx=start_idx,
                end_idx=end_idx,
                total_clips=support_size,
                min_frames_per_clip=min_frames_per_clip,
            )

        support_samples: list[ClipCaptionSample] = [
            build_clip_sample(
                base_sample=sample,
                clip_index=clip_idx,
                clip_start_idx=clip_start,
                clip_end_idx=clip_end,
                fps=fps,
            )
            for clip_idx, (clip_start, clip_end) in enumerate(support_ranges)
        ]
        for support_sample in support_samples:
            load_uniform_sample_frames(support_sample, min_frames_per_clip)
    except Exception:
        return None

    return TaskBatch(
        task_name=sample.sample_id.replace(":", "__"),
        task_mode=task_mode,
        task_samples=[sample],
        support_samples=support_samples,
        query_samples=[],
        source_distribution={sample.source: 1},
    )


# ---- soft-prompt captioning ----
@contextmanager
def soft_token_context(vlm_model, soft_tokens: torch.Tensor):
    """Replace the first n_soft embedding positions with Φ inside the language model forward."""
    n_soft = soft_tokens.shape[0]
    inner = vlm_model.model
    target = inner.language_model if hasattr(inner, "language_model") else inner

    def pre_hook(module, args, kwargs):
        embeds = kwargs.get("inputs_embeds")
        if embeds is None:
            input_ids = kwargs.get("input_ids")
            if kwargs.get("past_key_values") is not None and input_ids is not None and input_ids.shape[1] < n_soft:
                return args, kwargs
            embed_layer = getattr(module, "embed_tokens", None)
            if input_ids is None or embed_layer is None:
                return args, kwargs
            embeds = embed_layer(input_ids)
            kwargs["input_ids"] = None
        elif kwargs.get("past_key_values") is not None and embeds.shape[1] < n_soft:
            return args, kwargs
        if embeds.shape[1] < n_soft:
            return args, kwargs
        phi = soft_tokens.unsqueeze(0).expand(embeds.shape[0], -1, -1).to(device=embeds.device, dtype=embeds.dtype)
        kwargs["inputs_embeds"] = torch.cat([phi, embeds[:, n_soft:, :]], dim=1)
        return args, kwargs

    handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()

def _prepend_dummy_tokens(inputs: dict, n_soft: int, device, dummy_token_id: int = 0) -> dict:
    """Prepend n_soft placeholder tokens to the text-side sequence tensors (overwritten by Φ)."""
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    dummy_ids = torch.full((ids.shape[0], n_soft), dummy_token_id, dtype=ids.dtype, device=device)
    soft_mask = torch.ones(mask.shape[0], n_soft, dtype=mask.dtype, device=device)
    out = {
        **inputs,
        "input_ids": torch.cat([dummy_ids, ids], dim=1),
        "attention_mask": torch.cat([soft_mask, mask], dim=1),
    }
    for key in ("token_type_ids", "mm_token_type_ids"):
        token_type_ids = inputs.get(key)
        if token_type_ids is None or getattr(token_type_ids, "ndim", 0) != 2:
            continue
        if token_type_ids.shape[1] != ids.shape[1]:
            continue
        soft_types = torch.zeros(
            token_type_ids.shape[0],
            n_soft,
            dtype=token_type_ids.dtype,
            device=device,
        )
        out[key] = torch.cat([soft_types, token_type_ids], dim=1)
    return out

def _dummy_token_id(vlm_processor) -> int:
    """Return a harmless token id for reserving soft-token positions."""
    tokenizer = getattr(vlm_processor, "tokenizer", vlm_processor)
    for attr in ("pad_token_id", "eos_token_id", "bos_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            return int(value)
    return 0

def generate_caption_with_phi(
    *,
    vlm_model,
    vlm_processor,
    frames: list[Image.Image],
    prompt_text: str,
    soft_tokens: torch.Tensor,
    max_new_tokens: int,
) -> str:
    """Generate one greedy caption with Φ injected."""
    content = [{"type": "image"} for _ in frames]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    text = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = vlm_processor(text=[text], images=frames, padding=True, return_tensors="pt")
    device = next(vlm_model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    n_soft = soft_tokens.shape[0]
    inputs = _prepend_dummy_tokens(inputs, n_soft, device, _dummy_token_id(vlm_processor))
    input_seq_len = inputs["input_ids"].shape[1]  # = n_soft + original_L

    with soft_token_context(vlm_model, soft_tokens):
        with torch.no_grad():
            gen_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
    new_tokens = gen_ids[:, input_seq_len:]
    return vlm_processor.batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

# ---- captioning baseline (no soft prompt) ----
def qwen_caption(
    model: Any,
    processor: Any,
    frames: list[Image.Image],
    caption_instruction: str,
    max_new_tokens: int,
) -> str:
    """Plain VLM caption used by the paper's Captioning baseline (no soft prompt)."""
    content: list[dict] = [{"type": "image"} for _ in frames]
    content.append({"type": "text", "text": caption_instruction})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.2)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], output_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
