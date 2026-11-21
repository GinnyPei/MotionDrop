# MotionDrop

English | [简体中文](README_zh-CN.md)

MotionDrop is a training-free video token compression method: between the
vision encoder and the language model, it drops visual tokens that repeat the
previous frame. Below are the steps to reproduce the 1 fps results on OVO-Bench
(LLaVA-OneVision-7B, 90% and 95% compression).

## 1. Install

You need Linux, NVIDIA GPUs (we use two 12 GB RTX 3080 Ti), Python 3.10–3.12,
and 32 GB of RAM or more.

```bash
git clone https://github.com/GinnyPei/MotionDrop.git
cd MotionDrop
./scripts/setup.sh
.venv/bin/motiondrop doctor
```

## 2. Get the model and data

Download:

```bash
./scripts/download_models.sh --models llava-ov-7b
./scripts/download_datasets.sh --datasets ovobench
```

If you already have them, skip the download and create `.env` in the
repository root with their paths:

```bash
MOTIONDROP_MODELS_DIR=/data/models        # holds llava-ov-7b/ or llava-onevision-qwen2-7b-ov-hf/
MOTIONDROP_OVOBENCH_DIR=/data/ovobench    # holds ovo_bench_new.json and chunked_videos/
```

## 3. Run the evaluation

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./scripts/run_ovobench.sh \
  --fps 1 --max-frames 4096 --compression-rates 90,95 \
  --batch-size 1 --max-gpu-memory-gib 10 \
  --output results/ovobench-fps1/llava-ov-7b
```

- If a run is interrupted, run the same command again; finished questions are
  skipped.

## More

- All options: `.venv/bin/motiondrop ovobench --help`.
- StreamingBench: `./scripts/run_streamingbench.sh`.
- Streaming runtime and web demo: see
  [`streaming_runtime/README.md`](streaming_runtime/README.md) and
  [`streaming_demo/README.md`](streaming_demo/README.md).
- Method implementation: [`src/motiondrop/compression.py`](src/motiondrop/compression.py).

## License

[Apache License 2.0](LICENSE).
