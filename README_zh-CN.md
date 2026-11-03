# MotionDrop

[English](README.md) | 简体中文

MotionDrop 是一种无需训练、感知运动的视频大模型视觉 token 压缩方法。它位于视觉编码器
与语言模型之间：运动越多的帧分到越多的 token 预算；在每一帧内部，保留发生了变化、或被
编码器重点关注的 token。本仓库包含离线基准评测（OVO-Bench、StreamingBench）、按延迟
SLO 自动决定 token 预算的流水线式流式运行时，以及可交互的实时视频 Web 演示。

## 目录

- [方法原理](#方法原理)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [安装](#安装)
- [模型与数据集](#模型与数据集)
- [基准评测](#基准评测)
- [配置](#配置)
- [流式运行时](#流式运行时)
- [交互式 Web 演示](#交互式-web-演示)
- [仓库结构](#仓库结构)
- [开发](#开发)
- [许可证](#许可证)

## 方法原理

对于一段 `T` 帧、每帧 `N` 个视觉 token 的视频片段，在目标压缩率 `r` 下，MotionDrop
精确保留 `round((1 - r) · T · N)` 个 token（每帧至少保留 1 个）：

1. **运动残差。** 每个 token 与上一帧对应位置 3×3 邻域内最相似的 token 比较，残差 `D`
   为 1 减去该余弦相似度。局部匹配可以容忍镜头或物体的小幅位移。
2. **帧预算。** 帧的运动强度为 `E_t = mean(max(D - τ, 0))`，噪声阈值 `τ = 0.1`。整段
   预算按 `E_t` 的比例分配到各帧，并用最大余数法取整，总数精确等于目标值。首帧没有参考
   帧，取其余帧运动强度的均值。
3. **token 打分。** 令 `λ_t = E_t / (E_t + τ)`，每个 token 的得分为
   `λ_t · rank(D) + (1 - λ_t) · rank(spatial)`，其中空间分数是该 token 在视觉编码器中
   收到的平均注意力。因此静止帧按显著性排序，运动帧按变化程度排序。
4. **重复抑制。** 若一个静止 token（`D ≤ τ`）在上一帧匹配到的 token 已被保留，则把它排
   到最后，让预算留给模型尚未看到的内容。该步骤默认开启；`--no-dedup` 即论文中原始的
   方法。
5. 每帧保留得分最高的 token，并保持原有顺序。

实现见 [`src/motiondrop/compression.py`](src/motiondrop/compression.py)。

| 模型别名 | 权重 |
| --- | --- |
| `llava-ov-7b`（默认） | `llava-hf/llava-onevision-qwen2-7b-ov-hf` |
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B`（非思考模式） |

## 环境要求

- Linux、NVIDIA GPU 与 CUDA。7–9B 模型以半精度加载，并自动分布到所有可见 GPU 上；我们
  使用两张 RTX 3080 Ti（12 GB），并设置 `MOTIONDROP_MAX_GPU_MEMORY_GIB=10`。
- Python 3.10–3.12。依赖版本固定在 [`pyproject.toml`](pyproject.toml) 中
  （PyTorch ≥ 2.6、Transformers 5.2.0）。
- 足够存放模型权重和评测视频的磁盘空间；运行
  `./scripts/download_datasets.sh --dry-run` 可查看数据集大小。

## 快速开始

```bash
git clone https://github.com/GinnyPei/MotionDrop.git
cd MotionDrop
./scripts/setup.sh
./scripts/download_models.sh --models llava-ov-7b
./scripts/download_datasets.sh --datasets ovobench streamingbench
./scripts/run_ovobench.sh
./scripts/run_streamingbench.sh
```

以上命令在 LLaVA-OneVision-7B 上以 0/50/70/90/95% 压缩率评测 MotionDrop。每次运行
都会打印准确率汇总，并保存到 `results/<benchmark>/<model>/c<rate>/summary.json`。

- `./scripts/run_ovobench.sh --limit 4` 只跑 4 个单元，用于检查流程。
- `MOTIONDROP_MODEL=qwen3-vl-8b`（或 `qwen3.5-9b`）切换模型，需先下载对应权重。
- 权重和数据集放在其他位置时，把路径写进 `.env`（见[配置](#配置)）。

## 安装

```bash
./scripts/setup.sh          # 创建 .venv 并安装依赖
./scripts/setup.sh --dev    # 同时安装 lint 工具
```

`setup.sh` 不会下载模型或数据。检查环境：

```bash
.venv/bin/motiondrop doctor
```

## 模型与数据集

按需下载：

```bash
./scripts/download_models.sh --models llava-ov-7b
./scripts/download_datasets.sh --datasets ovobench streamingbench
```

- 两个脚本都支持自定义目录（`--models-dir /mnt/models`、`--data-dir /mnt/data`）。
- 下载中断后可续传。
- 其他常用参数：`--keep-archives`、`--max-workers N`、`--skip-disk-check`。
- 需要鉴权或镜像时，设置 `HF_TOKEN` 或 `HF_ENDPOINT`。

## 基准评测

评测采用因果输入：每个问题只能看到其时间戳之前的视频。默认设置如下：

| 设置 | 默认值 |
| --- | --- |
| 压缩率 | 0、50、70、90、95% |
| 帧数 | 先按 1 fps 采样，再均匀截取至每段最多 6 帧 |
| 帧面积 | 模型预处理前为 224² 到 384² 像素 |
| 批大小 | 每次模型调用 2 个视频 |

每个压缩率会在输出目录下写入 `c<rate>/units.jsonl`（逐题记录）、`c<rate>/results.json`
和 `c<rate>/summary.json`。重新运行同一命令时会跳过已完成的单元。

### OVO-Bench

```bash
./scripts/run_ovobench.sh
```

显式指定配置：

```bash
MOTIONDROP_MODEL=llava-ov-7b \
MOTIONDROP_COMPRESSION_RATES=90,95 \
MOTIONDROP_MAX_FRAMES=6 \
MOTIONDROP_MAX_GPU_MEMORY_GIB=10 \
  ./scripts/run_ovobench.sh
```

使用仓库外的模型和数据集：

```bash
MOTIONDROP_MODEL_PATH=/mnt/models/llava-ov-7b \
MOTIONDROP_OVO_ANNOTATIONS=/mnt/data/ovobench/ovo_bench_new.json \
MOTIONDROP_OVO_VIDEOS=/mnt/data/ovobench/chunked_videos \
MOTIONDROP_OUTPUT=results/ovobench/llava-ov-7b \
  ./scripts/run_ovobench.sh
```

用 `MOTIONDROP_OVO_TASKS`（例如 `"EPM ASI HLD"`）选择任务，用 `MOTIONDROP_OVO_FPS`
设置采样帧率。

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

用 `MOTIONDROP_STREAMING_DATASETS`（例如 `"realtime_visual_understanding"`）或
`MOTIONDROP_QUESTION_ID_FILE`（每行一个题目 ID）限定范围；
`MOTIONDROP_STREAMING_DATA_ROOT` 指定自定义数据目录。

## 配置

所有默认值都定义在 Python CLI 中，见 `.venv/bin/motiondrop <command> --help`。
`scripts/run_*.sh` 只转发你设置过的 `MOTIONDROP_*` 变量（映射见
[`scripts/lib.sh`](scripts/lib.sh)）；额外传入的参数直接交给 CLI，优先级最高：

```bash
./scripts/run_ovobench.sh --limit 8 --no-dedup
```

路径从环境变量读取，也可以写在仓库根目录的 `.env` 文件中（已被 git 忽略；环境变量优先）。
直接指向已有的数据和权重即可，无需移动文件或建立软链接：

```bash
# .env
MOTIONDROP_MODELS_DIR=/data/models                      # 其中有 llava-ov-7b/ 或 llava-onevision-qwen2-7b-ov-hf/
MOTIONDROP_OVOBENCH_DIR=/data/ovobench                  # 其中有 ovo_bench_new.json 和 chunked_videos/
MOTIONDROP_STREAMINGBENCH_DIR=/data/StreamingBench      # 其中有 StreamingBench/ 和 videos/
```

| 变量 | 默认值 | 内容 |
| --- | --- | --- |
| `MOTIONDROP_MODELS_DIR` | `checkpoints/` | 每个模型一个目录，以别名或 Hugging Face 仓库名命名 |
| `MOTIONDROP_DATA_DIR` | `data/` | 下载目标目录，也是两个数据集默认路径的上级目录 |
| `MOTIONDROP_OVOBENCH_DIR` | `data/ovobench/` | `ovo_bench_new.json`、`chunked_videos/` |
| `MOTIONDROP_STREAMINGBENCH_DIR` | `data/streamingbench/` | `StreamingBench/`（CSV）、`videos/` |
| `MOTIONDROP_RESULTS_DIR` | `results/` | 评测与运行时输出 |

`MOTIONDROP_MODEL_PATH`（或 `--model-path`）可直接指定单个权重目录。

MotionDrop 超参数：

| CLI 参数 | 环境变量 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--motion-noise-threshold` | `MOTIONDROP_MOTION_NOISE_THRESHOLD` | `0.1` | 残差噪声阈值 `τ` |
| `--motion-local-radius` | `MOTIONDROP_MOTION_LOCAL_RADIUS` | `1` | 匹配半径；1 即 3×3 邻域 |
| `--first-frame-policy` | `MOTIONDROP_FIRST_FRAME_POLICY` | `window_mean` | 首帧预算；`zero` 只给 1 个 token |
| `--budget-floor` | `MOTIONDROP_BUDGET_FLOOR` | `0.0` | 每帧保底获得均匀份额的比例 |
| `--motion-power` | `MOTIONDROP_MOTION_POWER` | `1.0` | 帧预算按 `E_t ** power` 的比例分配 |
| `--dedup` / `--no-dedup` | `MOTIONDROP_DEDUP`（`1`/`0`） | 开启 | 重复抑制（第 4 步） |

## 流式运行时

运行时把视频切成若干帧一组的窗口，在语言模型处理上一个窗口的同时，完成当前窗口的
解码、编码与压缩。每个窗口的 token 预算可以由拟合好的延迟模型和延迟 SLO 决定，也可以
固定。

```bash
./scripts/run_streaming_runtime.sh
```

默认使用一段 OVO-Bench 示例视频、仓库自带的延迟模型
[`streaming_runtime/profiles/llava-ov-7b.json`](streaming_runtime/profiles/llava-ov-7b.json)、
150 ms SLO、每窗口 4 帧、最多采样 8 帧，结果写入 `results/streaming-runtime/run.json`。

固定压缩率：

```bash
MOTIONDROP_STREAM_VIDEO=/path/to/video.mp4 \
MOTIONDROP_STREAM_QUERY="What is happening in the video?" \
MOTIONDROP_FIXED_COMPRESSION=0.9 \
  ./scripts/run_streaming_runtime.sh
```

由 SLO 控制预算：

```bash
MOTIONDROP_STREAM_VIDEO=/path/to/video.mp4 \
MOTIONDROP_SLO_MS=500 \
MOTIONDROP_WINDOW_FRAMES=4 \
MOTIONDROP_QUEUE_SIZE=2 \
  ./scripts/run_streaming_runtime.sh
```

自带的延迟模型是在 RTX 3080 Ti 上测得的。为你自己的硬件拟合一份：

```bash
.venv/bin/python scripts/fit_latency_model.py \
  --video /path/to/representative.mp4 \
  --frame-counts 1,2,4,6 \
  --budgets 32,64,128,256,512,768,1024 \
  --output streaming_runtime/profiles/llava-ov-7b.json
```

更多命令（包括在指定时间戳提问、实时多任务运行时）见
[`streaming_runtime/README.md`](streaming_runtime/README.md)。

## 交互式 Web 演示

演示一边播放视频，服务器一边把新到的帧持续压缩进同一份 KV 历史。除了一次性提问，还可以
下达长时任务，例如“解说这场比赛”（以实时字幕显示）或“有人进球时提醒我”（以提醒显示），
并在任务运行期间继续提问。

通过 SSH 私密访问：

```bash
./scripts/run_streaming_demo.sh                    # 在服务器上运行，绑定 127.0.0.1:7860
ssh -N -L 7860:127.0.0.1:7860 USER@SERVER_IP       # 在本地电脑上运行
```

然后打开 <http://127.0.0.1:7860>。

在可信网络中直接访问：

```bash
./scripts/run_streaming_demo_public.sh                               # 80 端口，需输入一次 sudo 密码
MOTIONDROP_DEMO_PUBLIC_PORT=8080 ./scripts/run_streaming_demo_public.sh  # 无需 sudo
```

只有端口代理以管理员权限运行，模型进程仍属于当前用户；`Ctrl+C` 会同时停止两者。演示没有
身份验证，请勿暴露在不可信网络中。详见 [`streaming_demo/README.md`](streaming_demo/README.md)。

## 仓库结构

```text
src/motiondrop/
  compression.py        MotionDrop token 选择
  adapters/             LLaVA-OneVision 与 Qwen 接入（视觉编码器之后、语言模型之前）
  benchmarks/           OVO-Bench 与 StreamingBench 评测
  streaming_runtime/    流水线运行时、SLO 控制器、实时多任务运行时、演示服务
scripts/                安装、下载与运行脚本
streaming_runtime/      延迟模型与运行时文档
streaming_demo/         Web 演示前端
```

## 开发

```bash
./scripts/setup.sh --dev
.venv/bin/python -m ruff check src
.venv/bin/python -m ruff format src
```

## 许可证

[Apache License 2.0](LICENSE)。
