# Streaming runtime commands

## One-command local run

The repository includes the fitted 4×RTX 3080 Ti latency profile. The launcher
finds the LLaVA-OV-7B checkpoint and an OVO-Bench sample clip through the paths
in `.env` (see the main README):

```bash
./scripts/run_streaming_runtime.sh
```

Override any default with the environment variables shown below when needed.

## Fixed compression

```bash
MOTIONDROP_STREAM_VIDEO=/path/to/video.mp4 \
MOTIONDROP_STREAM_QUERY="What is happening now?" \
MOTIONDROP_FIXED_COMPRESSION=0.9 \
  ./scripts/run_streaming_runtime.sh
```

## Fit a latency model

```bash
.venv/bin/python scripts/fit_latency_model.py \
  --video /path/to/representative.mp4 \
  --frame-counts 1,2,4,6 \
  --budgets 32,64,128,256,512,768,1024 \
  --warmup 2 \
  --repeats 5 \
  --samples-output streaming_runtime/profiles/llava-ov-7b.samples.json \
  --output streaming_runtime/profiles/llava-ov-7b.json
```

Fit again from existing samples without loading the model:

```bash
.venv/bin/python scripts/fit_latency_model.py \
  --samples streaming_runtime/profiles/llava-ov-7b.samples.json \
  --output streaming_runtime/profiles/llava-ov-7b.json
```

## SLO-controlled runtime

```bash
MOTIONDROP_STREAM_VIDEO=/path/to/video.mp4 \
MOTIONDROP_STREAM_QUERY="What is happening now?" \
MOTIONDROP_LATENCY_MODEL=streaming_runtime/profiles/llava-ov-7b.json \
MOTIONDROP_SLO_MS=500 \
MOTIONDROP_WINDOW_FRAMES=4 \
MOTIONDROP_QUEUE_SIZE=2 \
  ./scripts/run_streaming_runtime.sh
```

## Query at a stream timestamp

```bash
MOTIONDROP_STREAM_VIDEO=/path/to/video.mp4 \
MOTIONDROP_STREAM_QUERY="What happened before this point?" \
MOTIONDROP_QUERY_AT_SECONDS=120 \
  ./scripts/run_streaming_runtime.sh
```

## Live multi-task runtime

`motiondrop demo` runs on `LiveScheduler` (`src/motiondrop/streaming_runtime/live.py`),
which serves long-running tasks and instant questions on the same stream.

- **Trunk + branches.** The KV cache (the *trunk*) holds only the system
  prefix and video windows. Each task runs as a *branch* appended to the trunk
  and rolled back afterwards (`LlavaStreamingBackend.generate_branch` /
  `score_choices`), so commentary, alerts and questions never see each other's
  prompts or outputs. `--shared-history` / `commit_outputs=True` reproduces the
  old behaviour of writing every answer into the trunk.
- **One GPU owner.** A single worker thread owns the cache and picks work in
  this order: instant jobs (questions, intent routing) > window prefill >
  periodic tasks. A periodic task only starts when the trunk has caught up, and
  it is preempted between tokens when a question arrives (it then retries).
- **Task kinds.** *Standing* tasks (e.g. commentary) produce a line every
  `standing_interval_s`; near-duplicates of recent lines and `<SILENT>` are
  suppressed. *Watch* tasks score Yes/No every `watch_interval_s` and emit an
  alert when p(yes) ≥ `watch_threshold`, with a cooldown.
- **Memory.** Task outputs are written to a timestamped text log. Questions
  retrieve recent and lexically matching notes about video that has already
  been evicted (`memory_scope="evicted"`); while video is still resident the
  model reads the frames instead of its own, possibly wrong, notes
  (`memory_scope="all"` disables this gate). `memory_interval_s` adds a hidden
  note-taking task.
- **Eviction.** With `max_context_tokens`, the oldest whole windows are evicted
  (down to `evict_to_ratio` of the limit). In `contiguous` mode the surviving
  keys are re-rotated by the removed length so RoPE positions stay gap-free
  (exactly equivalent to a fresh prefill for a one-layer model); `absolute`
  keeps the original positions.
- **Time markers.** `time_markers=True` prefixes each window with `[12.0s]` so
  prompts can refer to "frames after [x s]".

### Evaluating long-running + instant tasks

```bash
.venv/bin/motiondrop live-eval --suite proactive --name watch-only
.venv/bin/motiondrop live-eval --suite instant --name commentary --commentary --per-video 5
.venv/bin/motiondrop live-eval --suite ovo-backward --name evict1024 --max-context-tokens 1024
.venv/bin/motiondrop live-eval --suite realtime --name commentary --commentary --limit 2
.venv/bin/python scripts/live_eval_report.py   # per-sample report: results/live-eval/REPORT.md
```

Datasets are read from `MOTIONDROP_STREAMINGBENCH_DIR` (CSV under
`StreamingBench/`, videos under `videos/`, including `videos/proactive_output`)
and `MOTIONDROP_OVOBENCH_DIR`. Use `--only` to run named streams and `--limit` for a prefix.
