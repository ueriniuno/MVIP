from __future__ import annotations

import argparse
import gc
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

THIS_FILE = Path(__file__).resolve()
RUNNERS_DIR = THIS_FILE.parent
PROJECT_ROOT = RUNNERS_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mvip.utils import (  # noqa: E402
    TaskBatch,
    _dummy_token_id,
    _prepend_dummy_tokens,
    read_json,
    write_json,
    load_uniform_sample_frames,
    resolve_gen_num_frames,
    run_state_path,
    soft_token_context,
    state_root_for_run,
    schedule_path_for_outer_step,
    load_task_schedule,
    save_task_schedule,
    order_history_entries,
)
from mvip.utils import (  # noqa: E402
    load_vlm_model,
    sample_generated_frames,
)
from mvip.utils import CaptionSample  # noqa: E402
from mvip.clip_score import clip_similarity, load_clip_model  # noqa: E402
from mvip.wan_diffusers import (  # noqa: E402
    generate_video_frames,
    load_wan_pipeline,
    normalize_num_frames,
    save_video,
)
from mvip.utils import build_support_only_task_from_sample  # noqa: E402

DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "train"
DEFAULT_PROMPT_PATH = RUNNERS_DIR / "prompts" / "initial_caption_prompt.txt"


_MANIFEST_VIDEO_LENGTHS: dict[str, int] = {}


# ── Caption sampling ──────────────────────────────────────────────────────────

def sample_captions(
    *,
    vlm_model,
    vlm_processor,
    frames: list[Image.Image],
    prompt_text: str,
    soft_tokens: torch.Tensor,
    n_captions: int,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """Sample N captions with temperature and Φ injected."""
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

    captions: list[str] = []
    with soft_token_context(vlm_model, soft_tokens):
        for _ in range(n_captions):
            with torch.no_grad():
                gen_ids = vlm_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                )
            new_tokens = gen_ids[:, input_seq_len:]
            caption = vlm_processor.batch_decode(
                new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            captions.append(caption)
    return captions


# ── Wan + CLIP scoring ────────────────────────────────────────────────────────

def init_phi_from_seed_text(
    *,
    vlm_model,
    vlm_processor,
    seed_text: str,
    n_soft_tokens: int,
    device: str,
) -> torch.Tensor:
    """Initialize soft tokens by averaging seed-text token embeddings."""
    tokenizer = getattr(vlm_processor, "tokenizer", vlm_processor)
    encoded = tokenizer(seed_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.get("input_ids")
    if input_ids is None or input_ids.numel() == 0:
        raise ValueError("Seed init text produced no tokens.")
    input_ids = input_ids.to(device)
    embed_table = vlm_model.get_input_embeddings()
    with torch.no_grad():
        token_embeds = embed_table(input_ids).squeeze(0)
    if token_embeds.ndim != 2 or token_embeds.shape[0] == 0:
        raise ValueError("Failed to obtain token embeddings for seed init text.")
    mean_embed = token_embeds.mean(dim=0, keepdim=True)
    return mean_embed.repeat(n_soft_tokens, 1).contiguous()


def load_manifest_samples(path: Path) -> list[CaptionSample]:
    payload = read_json(path)
    raw_items = payload.get("items") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise ValueError(f"Manifest has no item list: {path}")

    samples: list[CaptionSample] = []
    _MANIFEST_VIDEO_LENGTHS.clear()
    for idx, item in enumerate(raw_items, start=1):
        if not isinstance(item, dict):
            continue
        input_video = item.get("input_video")
        sample_id = item.get("sample_id")
        source = item.get("source")
        if not input_video or not sample_id or not source:
            continue
        chunk_start = item.get("chunk_start_sec")
        chunk_end = item.get("chunk_end_sec")
        duration = item.get("chunk_duration_sec")
        video_length = item.get("video_length")
        if video_length is not None:
            _MANIFEST_VIDEO_LENGTHS[str(sample_id)] = int(video_length)
        if duration is None and chunk_start is not None and chunk_end is not None:
            duration = float(chunk_end) - float(chunk_start)
        samples.append(
            CaptionSample(
                sample_id=str(sample_id),
                source=str(source),
                original_video_id=str(item.get("source_video_id") or Path(input_video).stem),
                segment_id=f"manifest_{idx:04d}",
                file_path=Path(input_video),
                duration=float(duration or 0.0),
                split="manifest",
                start_sec=None if chunk_start is None else float(chunk_start),
                end_sec=None if chunk_end is None else float(chunk_end),
            )
        )
    if not samples:
        raise ValueError(f"No usable samples loaded from manifest: {path}")
    return samples


def ordered_manifest_tasks_for_outer_step(
    *,
    samples: list[CaptionSample],
    outer_step: int,
    num_tasks: int,
    support_size: int,
    min_frames_per_clip: int,
) -> list[TaskBatch]:
    if not samples:
        return []
    task_batches: list[TaskBatch] = []
    start = (outer_step - 1) * max(1, num_tasks)
    for offset in range(max(1, num_tasks)):
        sample = samples[(start + offset) % len(samples)]
        task = build_support_only_task_from_sample(
            sample=sample,
            task_mode="same_sample_clip",
            support_size=support_size,
            min_frames_per_clip=min_frames_per_clip,
        )
        if task is None:
            raise RuntimeError(f"Manifest sample is not usable: {sample.sample_id} ({sample.file_path})")
        task_batches.append(task)
    return task_batches


def gen_num_frames_for_task(task_batch: TaskBatch, args: argparse.Namespace) -> int:
    if not getattr(args, "match_ref_num_frames", False):
        return int(args.num_gen_frames)
    if not task_batch.support_samples:
        return int(args.num_gen_frames)
    base_sample_id = getattr(task_batch.support_samples[0], "base_sample_id", None)
    if base_sample_id in _MANIFEST_VIDEO_LENGTHS:
        raw_frames = _MANIFEST_VIDEO_LENGTHS[base_sample_id]
        max_frames = int(getattr(args, "max_num_gen_frames", 0) or raw_frames)
        if max_frames > 0:
            raw_frames = min(raw_frames, max_frames)
        return normalize_num_frames(raw_frames, max_frames if max_frames > 0 else raw_frames)
    raw_frames = resolve_gen_num_frames(task_batch.support_samples[0], args)
    max_frames = int(getattr(args, "max_num_gen_frames", 0) or raw_frames)
    if max_frames > 0:
        raw_frames = min(raw_frames, max_frames)
    return normalize_num_frames(raw_frames, max_frames if max_frames > 0 else raw_frames)

def score_captions(
    *,
    captions: list[str],
    ref_frames: list[Image.Image],
    wan_pipe,
    wan_device: str,
    clip_model,
    clip_processor,
    clip_device: str,
    num_gen_frames: int,
    num_score_frames: int,
    wan_conditioning: str,
    seed: int,
    score_video_dir: Path | None = None,
) -> list[dict]:
    """Return list of dicts with 'score' and optional 'error' per caption."""
    results: list[dict] = []
    init_image = ref_frames[0] if wan_conditioning in {"first", "first_last"} and ref_frames else None
    last_image = ref_frames[-1] if wan_conditioning == "first_last" and ref_frames else None
    for cand_idx, caption in enumerate(captions):
        if not caption.strip():
            results.append({"score": 0.0, "error": "empty caption"})
            continue
        try:
            gen_frames = generate_video_frames(
                wan_pipe,
                wan_device,
                caption,
                seed,
                num_gen_frames,
                init_image=init_image,
                last_image=last_image,
            )
            ref_score_frames = sample_generated_frames(ref_frames, num_score_frames)
            gen_score_frames = sample_generated_frames(gen_frames, num_score_frames)
            score = float(clip_similarity(ref_score_frames, gen_score_frames, clip_model, clip_processor, clip_device))
            result = {"score": score, "wan_conditioning": wan_conditioning}
            if score_video_dir is not None:
                cand_dir = score_video_dir / f"candidate_{cand_idx:03d}"
                save_video(ref_frames, cand_dir / "reference.mp4", fps=24.0)
                save_video(gen_frames, cand_dir / "generated.mp4", fps=24.0)
                (cand_dir / "caption.txt").write_text(caption + "\n", encoding="utf-8")
                write_json(cand_dir / "sample_meta.json", {
                    "candidate_index": cand_idx,
                    "score": score,
                    "wan_conditioning": wan_conditioning,
                    "num_ref_frames": len(ref_frames),
                    "num_gen_frames": len(gen_frames),
                })
                result["score_video_dir"] = str(cand_dir)
            results.append(result)
        except Exception as exc:
            results.append({"score": 0.0, "error": str(exc)})
    return results


def score_video_dir_for_inner(task_dir: Path, step_idx: int) -> Path:
    outer_step_dir = task_dir.parent
    out_root = outer_step_dir.parent
    outer_suffix = outer_step_dir.name.rsplit("_", 1)[-1]
    return out_root / "score_videos" / f"outer_{outer_suffix}" / "videos" / task_dir.name / f"inner_{step_idx:03d}"


def select_best_scored_caption(
    captions: list[str], score_results: list[dict]
) -> tuple[int, str, float]:
    """Select the best successfully scored caption; stop the step if no candidate was scored."""
    if len(score_results) != len(captions):
        raise RuntimeError(
            "Best-of-N scoring returned a different number of results and captions "
            f"({len(score_results)} != {len(captions)}); refusing to update Phi."
        )

    valid: list[tuple[int, float]] = []
    failures: list[str] = []
    for idx, result in enumerate(score_results):
        error = result.get("error")
        try:
            score = float(result["score"])
        except (KeyError, TypeError, ValueError):
            error = error or "missing or invalid score"
            score = float("nan")
        if error or not math.isfinite(score):
            failures.append(f"candidate {idx}: {error or f'non-finite score {score!r}'}")
            continue
        valid.append((idx, score))

    if not valid:
        detail = "; ".join(failures[:3]) or "no score results"
        raise RuntimeError(
            "All Best-of-N candidates failed scoring; refusing to update Phi or "
            f"complete this training step. First errors: {detail}"
        )

    best_idx, best_score = max(valid, key=lambda pair: pair[1])
    return best_idx, captions[best_idx], best_score


def move_wan_pipeline_for_phase(wan_pipe, device: str, enabled: bool) -> None:
    """Move Wan only at phase boundaries to leave room for Qwen backward."""
    if not enabled or not torch.cuda.is_available():
        return
    current_device = getattr(wan_pipe, "_reptile_phase_device", None)
    if current_device == device:
        return
    print(f"    [mem] moving Wan pipeline to {device} for phase isolation", flush=True)
    wan_pipe.to(device)
    setattr(wan_pipe, "_reptile_phase_device", device)
    gc.collect()
    torch.cuda.empty_cache()


# ── CE loss with soft token injection ────────────────────────────────────────

def caption_likelihood_loss(
    *,
    vlm_model,
    vlm_processor,
    frames: list[Image.Image],
    prompt_text: str,
    soft_tokens: torch.Tensor,
    target_caption: str,
) -> torch.Tensor:
    """Compute -log p(target_caption | video, prompt_text + soft_tokens)."""
    device = next(vlm_model.parameters()).device
    n_soft = soft_tokens.shape[0]

    content = [{"type": "image"} for _ in frames]
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    prompt_text_str = vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    prompt_inputs = vlm_processor(text=[prompt_text_str], images=frames, padding=True, return_tensors="pt")
    prompt_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in prompt_inputs.items()}
    prompt_len = prompt_inputs["input_ids"].shape[1]

    full_text = prompt_text_str + " " + target_caption
    full_inputs = vlm_processor(text=[full_text], images=frames, padding=True, return_tensors="pt")
    full_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in full_inputs.items()}
    full_ids = full_inputs["input_ids"]  # [1, total_len]

    full_inputs = _prepend_dummy_tokens(full_inputs, n_soft, device, _dummy_token_id(vlm_processor))

    with soft_token_context(vlm_model, soft_tokens):
        outputs = vlm_model(**{k: v for k, v in full_inputs.items() if k != "labels"})

    # outputs.logits: [1, n_soft + total_len, vocab]; CE only over the caption positions.
    logit_start = n_soft + prompt_len - 1
    logit_end = n_soft + full_ids.shape[1] - 1
    caption_logits = outputs.logits[:, logit_start:logit_end, :].contiguous()
    caption_labels = full_ids[:, prompt_len:].contiguous()

    return F.cross_entropy(
        caption_logits.view(-1, caption_logits.shape[-1]),
        caption_labels.view(-1),
    )


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def make_inner_optimizer(phi_k: torch.Tensor, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.Adam([phi_k], lr=args.inner_lr)


def load_inner_checkpoint(
    *,
    phi_path: Path,
    optimizer_path: Path | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.optim.Optimizer]:
    phi_k = torch.load(phi_path, map_location=args.device).detach().requires_grad_(True)
    optimizer = make_inner_optimizer(phi_k, args)
    if optimizer_path is not None and optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=args.device))
    return phi_k, optimizer


def save_inner_checkpoint(
    *,
    phi_k: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    phi_path: Path,
    optimizer_path: Path,
) -> None:
    torch.save(phi_k.detach(), phi_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def compact_history_row(summary_payload: dict[str, Any]) -> dict[str, Any]:
    best_scores: list[float] = []
    ce_losses: list[float] = []
    for task_info in summary_payload.get("task_infos", []):
        inner_steps = task_info.get("inner_steps", [])
        if not inner_steps:
            continue
        last_step = inner_steps[-1]
        if "best_score" in last_step:
            best_scores.append(float(last_step["best_score"]))
        if "ce_loss" in last_step:
            ce_losses.append(float(last_step["ce_loss"]))
    row = {
        "outer_step": summary_payload["outer_step"],
        "algorithm": summary_payload["algorithm"],
        "phi_delta_norm": summary_payload["phi_delta_norm"],
        "phi_norm": summary_payload["phi_norm"],
        "phi_before_path": summary_payload.get("phi_before_path"),
        "phi_after_path": summary_payload.get("phi_after_path"),
        "task_count": len(summary_payload.get("task_infos", [])),
    }
    if best_scores:
        row["avg_final_best_score"] = sum(best_scores) / len(best_scores)
    if ce_losses:
        row["avg_final_ce_loss"] = sum(ce_losses) / len(ce_losses)
    return row


# ── Inner adaptation ──────────────────────────────────────────────────────────

def inner_adapt(
    *,
    phi: torch.Tensor,
    task_batch: TaskBatch,
    vlm_model,
    vlm_processor,
    clip_model,
    clip_processor,
    wan_pipe,
    wan_device: str,
    args: argparse.Namespace,
    step_dir: Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run inner adaptation for one task, return (Φ̃, info_dict)."""
    clip_device = next(iter(clip_model.parameters())).device

    phi_k = phi.clone().detach().requires_grad_(True)
    optimizer = make_inner_optimizer(phi_k, args)
    info: dict[str, Any] = {"inner_steps": [], "best_captions": []}

    for step_idx in range(1, args.inner_steps + 1):
        inner_dir = step_dir / f"inner_step_{step_idx:03d}"
        inner_dir.mkdir(parents=True, exist_ok=True)
        step_cache = inner_dir / "step_result.json"
        phi_before_path = inner_dir / "phi_before.pt"
        phi_after_path = inner_dir / "phi_after.pt"
        optimizer_state_path = inner_dir / "optimizer_state.pt"

        if args.resume and step_cache.exists() and phi_after_path.exists():
            step_result = read_json(step_cache)
            phi_k, optimizer = load_inner_checkpoint(
                phi_path=phi_after_path,
                optimizer_path=optimizer_state_path,
                args=args,
            )
            info["inner_steps"].append(step_result)
            if step_result.get("best_caption"):
                info["best_captions"].append(step_result["best_caption"])
            print(f"    [resume] inner step {step_idx}: loaded Φ_k checkpoint")
            continue

        torch.save(phi_k.detach(), phi_before_path)

        # Load frames directly from video file (ClipCaptionSample has no .frames field)
        ref_frames: list[Image.Image] = []
        for sup in task_batch.support_samples:
            try:
                frames = load_uniform_sample_frames(sup, args.num_caption_frames)
                ref_frames.extend(frames)
            except Exception:
                pass
        if not ref_frames:
            step_result = {"step": step_idx, "skipped": "no frames loaded"}
            write_json(step_cache, step_result)
            save_inner_checkpoint(
                phi_k=phi_k,
                optimizer=optimizer,
                phi_path=phi_after_path,
                optimizer_path=optimizer_state_path,
            )
            info["inner_steps"].append(step_result)
            continue
        sampled_frames = sample_generated_frames(ref_frames, args.num_caption_frames)

        if args.resume and step_cache.exists():
            step_result = read_json(step_cache)
            best_caption = step_result.get("best_caption", "")
            if best_caption:
                move_wan_pipeline_for_phase(
                    wan_pipe,
                    "cpu",
                    args.park_wan_for_vlm_backward,
                )
                optimizer.zero_grad()
                loss = caption_likelihood_loss(
                    vlm_model=vlm_model,
                    vlm_processor=vlm_processor,
                    frames=sampled_frames,
                    prompt_text=args.text_init,
                    soft_tokens=phi_k,
                    target_caption=best_caption,
                )
                loss.backward()
                optimizer.step()
            save_inner_checkpoint(
                phi_k=phi_k,
                optimizer=optimizer,
                phi_path=phi_after_path,
                optimizer_path=optimizer_state_path,
            )
            info["inner_steps"].append(step_result)
            if best_caption:
                info["best_captions"].append(best_caption)
            continue

        # Step 1: Sample N captions (detach phi_k; sampling is no-grad)
        captions_cache = inner_dir / "captions.json"
        if args.resume and captions_cache.exists():
            captions = read_json(captions_cache)
        else:
            captions = sample_captions(
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                frames=sampled_frames,
                prompt_text=args.text_init,
                soft_tokens=phi_k.detach(),
                n_captions=args.n_best_of,
                temperature=args.caption_temperature,
                max_new_tokens=args.max_new_tokens,
            )
            write_json(captions_cache, captions)

        # Step 2-3: Score via Wan+CLIP, select c*
        scores_cache = inner_dir / "scores.json"
        cached_scores = read_json(scores_cache) if args.resume and scores_cache.exists() else None
        if cached_scores is not None and all(r.get("error") for r in cached_scores):
            print(f"    [resume] ignoring cached all-failed scores: {scores_cache}")
            cached_scores = None
        if cached_scores is not None:
            score_results = cached_scores
        else:
            move_wan_pipeline_for_phase(
                wan_pipe,
                wan_device,
                args.park_wan_for_vlm_backward,
            )
            score_results = score_captions(
                captions=captions,
                ref_frames=ref_frames,
                wan_pipe=wan_pipe,
                wan_device=wan_device,
                clip_model=clip_model,
                clip_processor=clip_processor,
                clip_device=str(clip_device),
                num_gen_frames=gen_num_frames_for_task(task_batch, args),
                num_score_frames=args.num_caption_frames,
                wan_conditioning=args.wan_conditioning,
                seed=args.seed + step_idx,
                score_video_dir=score_video_dir_for_inner(step_dir, step_idx) if args.save_score_videos else None,
            )

        errors = [r.get("error") for r in score_results if r.get("error")]
        if errors:
            print(f"    [warn] {len(errors)} scoring error(s): {errors[0]}")

        # Stop if no candidate could be scored; cache scores only afterwards.
        best_idx, best_caption, best_score = select_best_scored_caption(
            captions, score_results
        )
        if cached_scores is None:
            write_json(scores_cache, score_results)

        # Step 4: CE loss + gradient update on phi_k
        loss_val = 0.0
        if best_caption:
            move_wan_pipeline_for_phase(
                wan_pipe,
                "cpu",
                args.park_wan_for_vlm_backward,
            )
            optimizer.zero_grad()
            loss = caption_likelihood_loss(
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                frames=sampled_frames,
                prompt_text=args.text_init,
                soft_tokens=phi_k,
                target_caption=best_caption,
            )
            loss.backward()
            optimizer.step()
            loss_val = float(loss.item())

        step_result = {
            "step": step_idx,
            "captions": captions,
            "score_results": score_results,
            "best_caption": best_caption,
            "best_score": best_score,
            "ce_loss": loss_val,
        }
        write_json(step_cache, step_result)
        save_inner_checkpoint(
            phi_k=phi_k,
            optimizer=optimizer,
            phi_path=phi_after_path,
            optimizer_path=optimizer_state_path,
        )
        info["inner_steps"].append(step_result)
        info["best_captions"].append(best_caption)
        print(f"    step {step_idx}: best_score={best_score:.4f} ce_loss={loss_val:.4f} c*={best_caption[:60]}")

    return phi_k.detach(), info


# ── Outer Reptile update ──────────────────────────────────────────────────────

def reptile_outer_update(
    phi: torch.Tensor,
    phi_tildes: list[torch.Tensor],
    epsilon: float,
) -> torch.Tensor:
    if not phi_tildes:
        return phi
    avg_delta = torch.stack([pt - phi for pt in phi_tildes]).mean(dim=0)
    return phi + epsilon * avg_delta


# ── Experiment name ───────────────────────────────────────────────────────────

def build_experiment_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return (
        f"reptile_token_bon"
        f"_ntok{args.n_soft_tokens}"
        f"_N{args.n_best_of}"
        f"_inner{args.inner_steps}"
        f"_outer{args.outer_steps}"
        f"_tasks{args.num_tasks_per_outer_step}"
        f"_lr{args.inner_lr}"
        f"_eps{args.epsilon}"
        f"_wan{args.wan_conditioning}"
        f"_seed{args.seed}"
    )


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Best-of-N Reptile with learnable soft tokens.")
    parser.add_argument("--task_manifest", type=Path, required=True, help="Ordered list of training clips (JSON).")
    parser.add_argument("--prompt_path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument(
        "--init_soft_prompt_text",
        type=str,
        default=None,
        help="If set, initialize Φ from mean embedding of this seed text (instead of random init).",
    )
    parser.add_argument("--n_soft_tokens", type=int, default=4)
    parser.add_argument("--out_root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--vlm_model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache_dir", type=Path, default=PROJECT_ROOT / ".cache")
    parser.add_argument("--n_best_of", type=int, default=5)
    parser.add_argument("--caption_temperature", type=float, default=0.8)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--num_caption_frames", type=int, default=4)
    parser.add_argument("--num_gen_frames", type=int, default=81)
    parser.add_argument("--match_ref_num_frames", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max_gen_seconds", type=float, default=0.0)
    parser.add_argument("--max_num_gen_frames", type=int, default=121)
    parser.add_argument(
        "--wan_conditioning",
        choices=["none", "first", "first_last"],
        default="none",
        help="Wan generation mode for scoring: text-only, first-frame I2V, or first+last-frame I2V.",
    )
    parser.add_argument(
        "--wan_cpu_offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Wan model CPU offload. Slower, but useful for first/first_last runs on crowded GPUs.",
    )
    parser.add_argument(
        "--park_wan_for_vlm_backward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Move Wan to CPU at Qwen backward boundaries, then restore it for scoring.",
    )
    parser.add_argument(
        "--save_score_videos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save candidate reference/generated videos for post-hoc all-metric scoring.",
    )
    parser.add_argument("--num_tasks_per_outer_step", type=int, default=1)
    parser.add_argument("--support_size", type=int, default=1)
    parser.add_argument("--min_frames_per_clip", type=int, default=4)
    parser.add_argument("--outer_steps", type=int, default=50)
    parser.add_argument("--inner_steps", type=int, default=2)
    parser.add_argument("--inner_lr", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--schedule_from_run_root",
        type=Path,
        default=None,
        help="Reuse outer-step task schedules from another run root for strict apples-to-apples comparisons.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    experiment_name = build_experiment_name(args)
    args.out_root.mkdir(parents=True, exist_ok=True)
    state_root = state_root_for_run(args.out_root)
    state_root.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    history_path = args.out_root / "history.json"
    phi_path = args.out_root / "learned_soft_tokens.pt"
    config_path = args.out_root / "run_config.json"
    current_run_state_path = run_state_path(args.out_root)

    write_json(config_path, {
        "algorithm": "reptile-token-bestofn",
        "experiment_name": experiment_name,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    })

    args.text_init = args.prompt_path.read_text(encoding="utf-8").strip()
    seed_text = None
    if args.init_soft_prompt_text is not None:
        seed_text = args.init_soft_prompt_text.strip()

    train_samples = load_manifest_samples(args.task_manifest)
    print(f"[*] Loaded ordered task manifest: {args.task_manifest}")

    # Load models
    vlm_model, vlm_processor = load_vlm_model(args.vlm_model_id, cache_dir=str(args.cache_dir))
    vlm_model.requires_grad_(False)
    clip_model, clip_processor = load_clip_model(args.device)
    wan_pipe, wan_device = load_wan_pipeline(
        cache_dir=str(args.cache_dir),
        image_to_video=args.wan_conditioning != "none",
        force_cpu_offload=args.wan_cpu_offload,
    )
    setattr(wan_pipe, "_reptile_phase_device", wan_device)
    move_wan_pipeline_for_phase(
        wan_pipe,
        "cpu",
        args.park_wan_for_vlm_backward,
    )

    # Infer hidden_dim from model (safe across VLM variants)
    hidden_dim = vlm_model.get_input_embeddings().embedding_dim

    print("[*] Algorithm: Best-of-N Reptile (soft tokens)")
    print(f"[*] n_soft_tokens={args.n_soft_tokens}, hidden_dim={hidden_dim}, N={args.n_best_of}")
    print(f"[*] inner_steps={args.inner_steps}, outer_steps={args.outer_steps}, ε={args.epsilon}")
    print(f"[*] Wan conditioning: {args.wan_conditioning} | CPU offload: {args.wan_cpu_offload}")
    print(f"[*] Park Wan for VLM backward: {args.park_wan_for_vlm_backward}")
    print(
        "[*] Wan frame length: "
        f"match_ref={args.match_ref_num_frames}, max_gen_seconds={args.max_gen_seconds:g}, "
        f"max_num_gen_frames={args.max_num_gen_frames}, fallback_num_gen_frames={args.num_gen_frames}"
    )
    print(f"[*] Experiment: {experiment_name} | Output: {args.out_root}")
    print(f"[*] Train samples: {len(train_samples)}")

    # Initialize Φ with small random values scaled to embedding std
    if args.resume and phi_path.exists():
        phi = torch.load(phi_path, map_location=args.device)
        print(f"[*] Resumed Φ from {phi_path}, shape={phi.shape}")
    else:
        if seed_text:
            phi = init_phi_from_seed_text(
                vlm_model=vlm_model,
                vlm_processor=vlm_processor,
                seed_text=seed_text,
                n_soft_tokens=args.n_soft_tokens,
                device=args.device,
            )
            print(
                f"[*] Initialized Φ: seed-text mean embedding [{args.n_soft_tokens} × {hidden_dim}] "
                f'from "{seed_text[:80]}"'
            )
        else:
            embed_std = vlm_model.get_input_embeddings().weight.data.std().item()
            phi = torch.randn(args.n_soft_tokens, hidden_dim, device=args.device) * (embed_std * 0.01)
            print(f"[*] Initialized Φ: random [{args.n_soft_tokens} × {hidden_dim}], "
                  f"std={embed_std * 0.01:.2e}")

    existing_history = read_json(history_path, default=[]) or []
    history_entries: dict[int, dict] = {
        int(item["outer_step"]): item
        for item in existing_history
        if isinstance(item, dict) and "outer_step" in item
    }
    write_json(current_run_state_path, {
        "state": "running",
        "outer_step": max(history_entries, default=0),
        "experiment_name": experiment_name,
    })

    for outer_step in range(1, args.outer_steps + 1):
        step_dir = args.out_root / f"outer_step_{outer_step:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        summary_path = step_dir / "outer_step_summary.json"
        phi_before_path = step_dir / "phi_before.pt"
        phi_after_path = step_dir / "phi_after.pt"

        if args.resume and summary_path.exists():
            payload = read_json(summary_path)
            saved_phi_path = payload.get("phi_after_path")
            if saved_phi_path and Path(saved_phi_path).exists():
                phi = torch.load(saved_phi_path, map_location=args.device)
            elif phi_after_path.exists():
                phi = torch.load(phi_after_path, map_location=args.device)
            elif payload.get("phi_after") is not None:
                phi = torch.tensor(payload["phi_after"], device=args.device)
            history_entries[outer_step] = compact_history_row(payload)
            print(f"[*] Resume: skipped outer step {outer_step}")
            continue

        print(f"\n{'='*60}")
        print(f"Outer step {outer_step}/{args.outer_steps}")
        print(f"{'='*60}")
        torch.save(phi.detach(), phi_before_path)

        schedule_path = schedule_path_for_outer_step(args.out_root, outer_step)
        if args.resume and schedule_path.exists():
            task_batches = load_task_schedule(schedule_path)
        else:
            reused = False
            if args.schedule_from_run_root is not None:
                source_schedule_path = schedule_path_for_outer_step(args.schedule_from_run_root, outer_step)
                if source_schedule_path.exists():
                    task_batches = load_task_schedule(source_schedule_path)
                    reused = True
                    print(f"[*] Reused schedule from {source_schedule_path}")
            if not reused:
                task_batches = ordered_manifest_tasks_for_outer_step(
                    samples=train_samples,
                    outer_step=outer_step,
                    num_tasks=args.num_tasks_per_outer_step,
                    support_size=args.support_size,
                    min_frames_per_clip=args.min_frames_per_clip,
                )
            save_task_schedule(schedule_path, task_batches)
        if not task_batches:
            raise RuntimeError("No valid tasks sampled.")

        phi_tildes: list[torch.Tensor] = []
        task_infos: list[dict] = []
        for task_batch in task_batches:
            print(f"  [task: {task_batch.task_name}] {task_batch.source_distribution}")
            task_dir = step_dir / task_batch.task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            task_phi_tilde_path = task_dir / "phi_tilde.pt"
            task_info_path = task_dir / "task_info.json"

            if args.resume and task_phi_tilde_path.exists() and task_info_path.exists():
                phi_tilde = torch.load(task_phi_tilde_path, map_location=args.device)
                info = read_json(task_info_path)
                print("    [resume] loaded task Φ̃ checkpoint")
            else:
                try:
                    phi_tilde, info = inner_adapt(
                        phi=phi,
                        task_batch=task_batch,
                        vlm_model=vlm_model,
                        vlm_processor=vlm_processor,
                        clip_model=clip_model,
                        clip_processor=clip_processor,
                        wan_pipe=wan_pipe,
                        wan_device=wan_device,
                        args=args,
                        step_dir=task_dir,
                    )
                except Exception as exc:
                    write_json(current_run_state_path, {
                        "state": "failed",
                        "outer_step": outer_step,
                        "task_name": task_batch.task_name,
                        "experiment_name": experiment_name,
                        "error": str(exc),
                    })
                    raise
                write_json(task_info_path, info)
                torch.save(phi_tilde.detach(), task_phi_tilde_path)
            phi_tildes.append(phi_tilde)
            task_infos.append(
                {
                    "task_name": task_batch.task_name,
                    "phi_tilde_path": str(task_phi_tilde_path),
                    "task_info_path": str(task_info_path),
                    **info,
                }
            )

        phi_before = phi.clone()
        phi = reptile_outer_update(phi, phi_tildes, args.epsilon)
        phi_delta_norm = float((phi - phi_before).norm().item())
        phi_norm = float(phi.norm().item())
        print(f"  Φ update norm: {phi_delta_norm:.6f} | Φ norm: {phi_norm:.6f}")

        torch.save(phi.detach(), phi_after_path)
        torch.save(phi, phi_path)

        summary_payload = {
            "outer_step": outer_step,
            "algorithm": "reptile-token-bestofn",
            "phi_delta_norm": phi_delta_norm,
            "phi_norm": phi_norm,
            "phi_before_path": str(phi_before_path),
            "phi_after_path": str(phi_after_path),
            "latest_phi_path": str(phi_path),
            "task_infos": task_infos,
        }
        write_json(summary_path, summary_payload)
        history_entries[outer_step] = compact_history_row(summary_payload)
        write_json(history_path, order_history_entries(history_entries))
        write_json(current_run_state_path, {
            "state": "outer_step_completed",
            "outer_step": outer_step,
            "experiment_name": experiment_name,
        })

    torch.save(phi, phi_path)
    write_json(current_run_state_path, {
        "state": "completed",
        "outer_step": args.outer_steps,
        "experiment_name": experiment_name,
    })
    print(f"\n[*] Learned soft tokens saved to: {phi_path}")


if __name__ == "__main__":
    main()
