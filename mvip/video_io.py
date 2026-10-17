from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image

CAPTION_INSTRUCTION = "Describe this video in detail."

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass
class ManifestItem:
    source: str
    sample_id: str
    input_video: str
    chunk_start_sec: float
    chunk_end_sec: float | None
    chunk_duration_sec: float | None
    chunk_index: int
    output_video: str
    reference_video: str
    first_frame: str
    last_frame: str
    caption_path: str
    metadata_path: str
    caption: str
    seed: int
    video_length: int
    fps: int
    input_fps: float | None
    input_frame_count: int | None


def get_video_info(path: Path) -> tuple[float | None, float | None, int | None]:
    """Return (duration_sec, fps, frame_count)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if fps > 0 and fc > 0:
                return fc / fps, fps, fc
        cap.release()
    except Exception:
        pass
    try:
        reader = iio.get_reader(str(path))
        meta = reader.get_meta_data()
        reader.close()
        fps = meta.get("fps")
        dur = meta.get("duration")
        if fps and dur:
            fc = max(1, int(round(float(fps) * float(dur))))
            return float(dur), float(fps), fc
    except Exception:
        pass
    return None, None, None


def video_length_for_duration(duration_sec: float | None, fps: int, fallback: int) -> int:
    if duration_sec is None or duration_sec <= 0:
        return fallback
    raw = max(1, int(math.ceil(duration_sec * fps)))
    if raw == 1:
        return 1
    return int(math.ceil((raw - 1) / 4) * 4 + 1)


def seconds_label(v: float | None) -> str:
    return "end" if v is None else f"{v:07.2f}".replace(".", "p")


def chunk_label(idx: int, start: float, end: float | None) -> str:
    return f"chunk_{idx:03d}_s{seconds_label(start)}_e{seconds_label(end)}"


def sample_frames(video_path: Path, max_frames: int) -> list[Image.Image]:
    reader = iio.get_reader(str(video_path))
    try:
        count = reader.count_frames()
        if count <= 0:
            raw = list(reader)
            count = len(raw)
            if count == 0:
                return []
            indices = np.linspace(0, count - 1, min(max_frames, count), dtype=int)
            return [Image.fromarray(raw[int(i)]).convert("RGB") for i in indices]
        indices = np.linspace(0, count - 1, min(max_frames, count), dtype=int)
        frames: list[Image.Image] = []
        for index in indices:
            try:
                frames.append(Image.fromarray(reader.get_data(int(index))).convert("RGB"))
            except IndexError:
                break
        if frames:
            return frames

        reader.close()
        raw = list(iio.get_reader(str(video_path)))
        count = len(raw)
        if count == 0:
            return []
        indices = np.linspace(0, count - 1, min(max_frames, count), dtype=int)
        return [Image.fromarray(raw[int(i)]).convert("RGB") for i in indices]
    finally:
        try:
            reader.close()
        except Exception:
            pass


def read_frame_at(path: Path, frame_index: int) -> Image.Image:
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = max(0, min(fc - 1, frame_index)) if fc > 0 else frame_index
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:
        pass
    reader = iio.get_reader(str(path))
    try:
        return Image.fromarray(reader.get_data(frame_index)).convert("RGB")
    finally:
        reader.close()


def extract_boundary_frames(
    video_path: Path,
    chunk_start_sec: float,
    chunk_end_sec: float | None,
    input_fps: float,
    frame_count: int,
    first_path: Path,
    last_path: Path,
) -> None:
    first_path.parent.mkdir(parents=True, exist_ok=True)
    start_idx = max(0, int(math.floor(chunk_start_sec * input_fps)))
    if chunk_end_sec is None:
        last_idx = frame_count - 1
    else:
        last_idx = min(frame_count - 1, max(start_idx, int(math.ceil(chunk_end_sec * input_fps)) - 1))
    if not first_path.exists():
        read_frame_at(video_path, start_idx).save(first_path)
    if not last_path.exists():
        read_frame_at(video_path, last_idx).save(last_path)


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def save_reference_chunk(
    input_video: str,
    chunk_start_sec: float,
    chunk_duration_sec: float | None,
    reference_path: Path,
) -> None:
    if reference_path.exists():
        return
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_video, "-ss", f"{chunk_start_sec:.6f}",
    ]
    if chunk_duration_sec is not None:
        cmd += ["-t", f"{chunk_duration_sec:.6f}"]
    cmd += ["-map", "0:v:0", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(reference_path)]
    subprocess.run(cmd, check=True)
