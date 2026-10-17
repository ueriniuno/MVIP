# MVIP: Meta Inverse Prompting for Video Generative Models (NeurIPS 2026)

Official implementation. [Project page](https://ueriniuno.github.io/mvip-page/)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Inference uses VideoX-Fun with the Wan2.2-Fun-5B-InP weights:

```bash
cd external
git clone https://github.com/aigc-apps/VideoX-Fun.git
git -C VideoX-Fun checkout 4a86483cc2fca7bd146108382a139e1ddc4ecdbc
git -C VideoX-Fun apply ../videox_fun.patch
huggingface-cli download alibaba-pai/Wan2.2-Fun-5B-InP --revision d3e911d57ec9e44f1f06dc565387afbe8d28f25f --local-dir Wan2.2-Fun-5B-InP
cd ..
```

Qwen2.5-VL-7B, Wan2.2-TI2V-5B (training) and the metric models are downloaded
from the Hugging Face Hub on first use. External code and model weights keep
their own licenses.

## Execution

### Inference

```bash
bash run_infer.sh test_videos/horsejump_high.mp4
bash run_infer.sh /path/to/video.mp4 outputs/my_video
```

Writes `videos/user/<video>/<chunk>/` with `caption.txt`, `generated.mp4`,
`reference.mp4`, the first/last frame, and `metadata.json`. The bundled test
clip is the `horsejump-high` sequence of DAVIS 2017.

### Data

```text
data/
├── ActivityNet/videos/<youtube_id>.mp4
├── MSR-VTT/videos/video<id>.mp4
└── VidProM/{VidProM_unique.csv, pika_videos/pika_videos_all, ms_videos/ms_videos_all, vc2_videos/vc2_videos_all}/
```

ActivityNet videos are downloaded from YouTube (`yt-dlp -f "b[ext=mp4]"`) for the
ids in `manifests/`, MSR-VTT videos come from the standard MSRVTT archive, and
`VidProM_unique.csv` and the `pika`/`ms`/`vc2` video archives from
[VidProM](https://huggingface.co/datasets/WenhaoWang/VidProM) are extracted into
`<source>_videos/<source>_videos_all/`. The remaining inputs are derived:

```bash
python scripts/prepare_data.py --data-root /path/to/data
```

This cuts the ActivityNet/MSR-VTT scene segments listed in
`manifests/eval_segments.json` into `<dataset>/segments/<video_id>/seg_NNN.mp4`
and generates the Wan2.2-TI2V-5B clips from the VidProM prompts (704x1280,
30 steps, seed 42) into `VidProM/wan_{81f,121f}/<index>_<uuid>/seed_42.mp4`.

### Evaluation

```bash
bash run_eval.sh /path/to/data
bash run_eval.sh /path/to/data outputs/captioning --baseline captioning
bash run_eval.sh /path/to/data outputs/flf_only --baseline flf-only
python scripts/aggregate_main_table.py --check
```

`run_eval.sh` generates a video for every clip in the manifest and writes the
metrics to `metrics_summary.csv`. `--baseline captioning` and `--baseline flf-only`
run the two baselines. `scripts/aggregate_main_table.py` rebuilds the main table
from `results/per_sample_metrics.csv`.

### Training

```bash
bash run_train.sh /path/to/data
bash run_train.sh /path/to/data outputs/train --resume
bash run_train.sh /path/to/data outputs/random_ntok4 --config configs/ablations/random_ntok4.yaml
```

`configs/ablations/` holds the Table 9 variants (random initialization, 4 and 8
soft tokens, K=3); each inherits `configs/train.yaml` and overrides one or two keys.
