# MotionDrop

English | [简体中文](README_zh-CN.md)

MotionDrop is a training-free, motion-aware visual token compression method for
video-language models. It runs between the vision encoder and the language
model: frames with more motion get a larger share of the token budget, and
inside each frame the tokens that changed or that the encoder attends to are
kept. The repository covers offline benchmarks (OVO-Bench, StreamingBench), a
pipelined streaming runtime with an SLO-driven token budget, and an
interactive live-video web demo.

## Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Installation](#installation)
- [Models and datasets](#models-and-datasets)
- [Benchmark evaluation](#benchmark-evaluation)
- [Configuration](#configuration)
- [Streaming runtime](#streaming-runtime)
- [Interactive web demo](#interactive-web-demo)
- [Repository layout](#repository-layout)
- [Development](#development)
- [License](#license)

## How it works

For a clip of `T` frames with `N` visual tokens each and a target compression
rate `r`, MotionDrop keeps exactly `round((1 - r) · T · N)` tokens (at least
one per frame):

1. **Motion residual.** Each token is compared with the most similar token in
   a 3×3 neighborhood of the previous frame; its residual `D` is one minus that
   cosine similarity. Local matching tolerates small camera or object shifts.
2. **Frame budget.** A frame's motion magnitude is
   `E_t = mean(max(D - τ, 0))` with noise threshold `τ = 0.1`. The clip budget
   is split across frames in proportion to `E_t` and rounded with the largest
   remainder, so the total is exact. The first frame has no reference and gets
   the mean magnitude of the other frames.
3. **Token score.** With `λ_t = E_t / (E_t + τ)`, each token is scored
   `λ_t · rank(D) + (1 - λ_t) · rank(spatial)`, where the spatial score is the
   mean attention the token receives in the vision encoder. Static frames are
   therefore ranked by saliency and moving frames by change.
4. **Duplicate suppression.** A static token (`D ≤ τ`) whose matched token in
   the previous frame was already kept is ranked last, so the budget goes to
   content the model has not seen yet. This is on by default; `--no-dedup`
   gives the method exactly as written in the paper.
5. Each frame keeps its top-scoring tokens in their original order.

The implementation is [`src/motiondrop/compression.py`](src/motiondrop/compression.py).

| Model alias | Checkpoint |
| --- | --- |
| `llava-ov-7b` (default) | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` (non-thinking mode) |

## Requirements

- Linux with NVIDIA GPUs and CUDA. The 7–9B models load in half precision and
  are spread across all visible GPUs; we use two RTX 3080 Ti (12 GB) cards
  with `MOTIONDROP_MAX_GPU_MEMORY_GIB=10`.
- Python 3.10–3.12. Dependencies are pinned in [`pyproject.toml`](pyproject.toml)
  (PyTorch ≥ 2.6, Transformers 5.2.0).
- Disk space for the checkpoints and benchmark videos; run
  `./scripts/download_datasets.sh --dry-run` to see the dataset sizes.

## Quick start

```bash
git clone https://github.com/GinnyPei/MotionDrop.git
cd MotionDrop
./scripts/setup.sh
./scripts/download_models.sh --models llava-ov-7b
./scripts/download_datasets.sh --datasets ovobench streamingbench
./scripts/run_ovobench.sh
./scripts/run_streamingbench.sh
```

This evaluates MotionDrop on LLaVA-OneVision-7B at 0/50/70/90/95% compression.
Each run prints its accuracy summary and saves it to
`results/<benchmark>/<model>/c<rate>/summary.json`.

- `./scripts/run_ovobench.sh --limit 4` runs four units as a pipeline check.
- `MOTIONDROP_MODEL=qwen3-vl-8b` (or `qwen3.5-9b`) switches the backbone;
  download its checkpoint first.
- To use checkpoints and datasets stored elsewhere, list their paths in `.env`
  (see [Configuration](#configuration)).

## Installation

```bash
./scripts/setup.sh          # create .venv and install dependencies
./scripts/setup.sh --dev    # also install the linter
```

`setup.sh` does not download models or data. Check the environment with:

```bash
.venv/bin/motiondrop doctor
```

## Models and datasets

Download only what you need:

```bash
./scripts/download_models.sh --models llava-ov-7b
./scripts/download_datasets.sh --datasets ovobench streamingbench
```

- Both scripts accept a custom location (`--models-dir /mnt/models`,
  `--data-dir /mnt/data`).
- Interrupted downloads resume.
- Other useful options: `--keep-archives`, `--max-workers N`, `--skip-disk-check`.
- For gated access or a mirror, set `HF_TOKEN` or `HF_ENDPOINT`.

## Benchmark evaluation

The runners use causal inputs: every question sees only the video up to its
timestamp. The defaults are:

| Setting | Default |
| --- | --- |
| Compression rates | 0, 50, 70, 90, 95% |
| Frames | sampled at 1 fps, then capped uniformly at 6 per clip |
| Frame area | 224² to 384² pixels before model preprocessing |
| Batch size | 2 videos per model call |

Each rate writes `c<rate>/units.jsonl` (per-question records),
`c<rate>/results.json`, and `c<rate>/summary.json` under the output directory.
Re-running a command skips the units that are already finished.

### OVO-Bench

```bash
./scripts/run_ovobench.sh
```

An explicit configuration:

```bash
MOTIONDROP_MODEL=llava-ov-7b \
MOTIONDROP_COMPRESSION_RATES=90,95 \
MOTIONDROP_MAX_FRAMES=6 \
MOTIONDROP_MAX_GPU_MEMORY_GIB=10 \
  ./scripts/run_ovobench.sh
```

A model and dataset stored outside the repository:

```bash
MOTIONDROP_MODEL_PATH=/mnt/models/llava-ov-7b \
MOTIONDROP_OVO_ANNOTATIONS=/mnt/data/ovobench/ovo_bench_new.json \
MOTIONDROP_OVO_VIDEOS=/mnt/data/ovobench/chunked_videos \
MOTIONDROP_OUTPUT=results/ovobench/llava-ov-7b \
  ./scripts/run_ovobench.sh
```

Choose tasks with `MOTIONDROP_OVO_TASKS` (for example `"EPM ASI HLD"`) and the
sampling rate with `MOTIONDROP_OVO_FPS`.

### StreamingBench

```bash
./scripts/run_streamingbench.sh
```

```bash
MOTIONDROP_MODEL=qwen3.5-9b \
MOTIONDROP_COMPRESSION_RATES=90 \
MOTIONDROP_BATCH_SIZE=1 \
MOTIONDROP_MAX_FRAMES=12 \
  ./scripts/run_streamingbench.sh
```

Restrict the run with `MOTIONDROP_STREAMING_DATASETS` (for example
`"realtime_visual_understanding"`) or `MOTIONDROP_QUESTION_ID_FILE` (one
question ID per line). `MOTIONDROP_STREAMING_DATA_ROOT` points at a custom
dataset directory.

## Configuration

All defaults live in the Python CLI; see `.venv/bin/motiondrop <command> --help`.
The `scripts/run_*.sh` wrappers only forward the `MOTIONDROP_*` variables you
set (the mapping is in [`scripts/lib.sh`](scripts/lib.sh)), and extra
arguments go straight to the CLI and take precedence:

```bash
./scripts/run_ovobench.sh --limit 8 --no-dedup
```

Paths are resolved from the environment or, more conveniently, from a `.env`
file in the repository root (git-ignored; the environment takes precedence).
Point them at existing copies instead of moving or linking files:

```bash
# .env
MOTIONDROP_MODELS_DIR=/data/models                      # holds llava-ov-7b/ or llava-onevision-qwen2-7b-ov-hf/
MOTIONDROP_OVOBENCH_DIR=/data/ovobench                  # holds ovo_bench_new.json and chunked_videos/
MOTIONDROP_STREAMINGBENCH_DIR=/data/StreamingBench      # holds StreamingBench/ and videos/
```

| Variable | Default | Contents |
| --- | --- | --- |
| `MOTIONDROP_MODELS_DIR` | `checkpoints/` | One directory per model, named by alias or by Hugging Face repo name |
| `MOTIONDROP_DATA_DIR` | `data/` | Download target and parent of the two dataset defaults |
| `MOTIONDROP_OVOBENCH_DIR` | `data/ovobench/` | `ovo_bench_new.json`, `chunked_videos/` |
| `MOTIONDROP_STREAMINGBENCH_DIR` | `data/streamingbench/` | `StreamingBench/` (CSV), `videos/` |
| `MOTIONDROP_RESULTS_DIR` | `results/` | Benchmark and runtime outputs |

`MOTIONDROP_MODEL_PATH` (or `--model-path`) selects one checkpoint directly.

MotionDrop hyperparameters:

| CLI flag | Variable | Default | Meaning |
| --- | --- | --- | --- |
| `--motion-noise-threshold` | `MOTIONDROP_MOTION_NOISE_THRESHOLD` | `0.1` | Residual noise threshold `τ` |
| `--motion-local-radius` | `MOTIONDROP_MOTION_LOCAL_RADIUS` | `1` | Matching radius; 1 is a 3×3 neighborhood |
| `--first-frame-policy` | `MOTIONDROP_FIRST_FRAME_POLICY` | `window_mean` | First-frame budget; `zero` gives it one token |
| `--budget-floor` | `MOTIONDROP_BUDGET_FLOOR` | `0.0` | Guaranteed fraction of the uniform per-frame share |
| `--motion-power` | `MOTIONDROP_MOTION_POWER` | `1.0` | Frame budgets are proportional to `E_t ** power` |
| `--dedup` / `--no-dedup` | `MOTIONDROP_DEDUP` (`1`/`0`) | on | Duplicate suppression (step 4) |

## Streaming runtime

The runtime decodes, encodes, and compresses a video in windows of frames
while the language model consumes the previous window. The token budget of
each window either comes from a fitted latency model and a latency SLO, or is
fixed.

```bash
./scripts/run_streaming_runtime.sh
```

The default run uses a sample OVO-Bench video, the bundled latency profile
[`streaming_runtime/profiles/llava-ov-7b.json`](streaming_runtime/profiles/llava-ov-7b.json),
a 150 ms SLO, four-frame windows, and at most eight sampled frames. It writes
`results/streaming-runtime/run.json`.

A fixed compression rate:

```bash
MOTIONDROP_STREAM_VIDEO=/path/to/video.mp4 \
MOTIONDROP_STREAM_QUERY="What is happening in the video?" \
MOTIONDROP_FIXED_COMPRESSION=0.9 \
  ./scripts/run_streaming_runtime.sh
```

An SLO-controlled budget:

```bash
MOTIONDROP_STREAM_VIDEO=/path/to/video.mp4 \
MOTIONDROP_SLO_MS=500 \
MOTIONDROP_WINDOW_FRAMES=4 \
MOTIONDROP_QUEUE_SIZE=2 \
  ./scripts/run_streaming_runtime.sh
```

The bundled profile was measured on RTX 3080 Ti GPUs. Fit one for your own
hardware:

```bash
.venv/bin/python scripts/fit_latency_model.py \
  --video /path/to/representative.mp4 \
  --frame-counts 1,2,4,6 \
  --budgets 32,64,128,256,512,768,1024 \
  --output streaming_runtime/profiles/llava-ov-7b.json
```

More commands, including querying at a stream timestamp and the live
multi-task runtime, are in [`streaming_runtime/README.md`](streaming_runtime/README.md).

## Interactive web demo

The demo plays a video while the server keeps compressing incoming frames into
one KV history. Besides one-off questions, it accepts long-running tasks such
as "commentate this match" (shown as live subtitles) or "tell me when someone
scores" (shown as alerts), and you can ask questions while they run.

Private access over SSH:

```bash
./scripts/run_streaming_demo.sh                    # on the server, binds 127.0.0.1:7860
ssh -N -L 7860:127.0.0.1:7860 USER@SERVER_IP       # on your laptop
```

Then open <http://127.0.0.1:7860>.

Direct access on a trusted network:

```bash
./scripts/run_streaming_demo_public.sh                               # port 80, asks for sudo once
MOTIONDROP_DEMO_PUBLIC_PORT=8080 ./scripts/run_streaming_demo_public.sh  # no sudo
```

Only the port proxy runs with administrator rights; the model stays under the
current user, and `Ctrl+C` stops both. The demo has no authentication, so do
not expose it on an untrusted network. See
[`streaming_demo/README.md`](streaming_demo/README.md) for details.

## Repository layout

```text
src/motiondrop/
  compression.py        MotionDrop token selection
  adapters/             LLaVA-OneVision and Qwen integration (post-encoder, pre-LLM)
  benchmarks/           OVO-Bench and StreamingBench runners
  streaming_runtime/    Pipelined runtime, SLO controller, live multi-task runtime, demo server
scripts/                Setup, download, and run wrappers
streaming_runtime/      Latency profiles and runtime docs
streaming_demo/         Web demo front end
```

## Development

```bash
./scripts/setup.sh --dev
.venv/bin/python -m ruff check src
.venv/bin/python -m ruff format src
```

## License

[Apache License 2.0](LICENSE).
