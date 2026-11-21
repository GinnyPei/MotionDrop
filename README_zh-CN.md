# MotionDrop

[English](README.md) | 简体中文

MotionDrop 是一种无需训练的视频 token 压缩方法：在视觉编码器和语言模型之间，删掉和上一帧
重复的视觉 token。下面是在 OVO-Bench 上复现 1 fps 结果的步骤（LLaVA-OneVision-7B，90% 和 95% 压缩）。

## 1. 安装

需要 Linux、NVIDIA GPU（我们用两张 12 GB 的 RTX 3080 Ti）、Python 3.10–3.12 和 32 GB 以上内存。

```bash
git clone https://github.com/GinnyPei/MotionDrop.git
cd MotionDrop
./scripts/setup.sh
.venv/bin/motiondrop doctor
```

## 2. 准备模型和数据

下载：

```bash
./scripts/download_models.sh --models llava-ov-7b
./scripts/download_datasets.sh --datasets ovobench
```

已经有的话不用下载，在仓库根目录新建 `.env` 写上路径：

```bash
MOTIONDROP_MODELS_DIR=/data/models        # 其中有 llava-ov-7b/ 或 llava-onevision-qwen2-7b-ov-hf/
MOTIONDROP_OVOBENCH_DIR=/data/ovobench    # 其中有 ovo_bench_new.json 和 chunked_videos/
```

## 3. 运行

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ./scripts/run_ovobench.sh \
  --fps 1 --max-frames 4096 --compression-rates 90,95 \
  --batch-size 1 --max-gpu-memory-gib 10 \
  --output results/ovobench-fps1/llava-ov-7b
```

- 中断后重新执行同一条命令，已完成的题会跳过。

## 其他

- 全部参数：`.venv/bin/motiondrop ovobench --help`。
- StreamingBench：`./scripts/run_streamingbench.sh`。
- 流式运行时和 Web 演示：见 [`streaming_runtime/README.md`](streaming_runtime/README.md) 和
  [`streaming_demo/README.md`](streaming_demo/README.md)。
- 方法实现：[`src/motiondrop/compression.py`](src/motiondrop/compression.py)。

## 许可证

[Apache License 2.0](LICENSE)。
