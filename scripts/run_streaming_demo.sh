#!/usr/bin/env bash
# Serve the interactive streaming demo on loopback. Defaults are those of
# `motiondrop demo --help`; set MOTIONDROP_LATENCY_MODEL to use SLO budgets.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
require_cli

ARGS=()
forward MOTIONDROP_STREAM_VIDEO --video
forward MOTIONDROP_MODEL_PATH --model-path
forward MOTIONDROP_LATENCY_MODEL --latency-model
forward MOTIONDROP_SLO_MS --slo-ms
forward MOTIONDROP_FIXED_COMPRESSION --fixed-compression
forward MOTIONDROP_DEMO_HOST --host
forward MOTIONDROP_DEMO_PORT --port
forward MOTIONDROP_STREAM_FPS --fps
forward MOTIONDROP_MAX_STREAM_FRAMES --max-stream-frames
forward MOTIONDROP_WINDOW_FRAMES --window-frames
forward MOTIONDROP_QUEUE_SIZE --queue-size
forward MOTIONDROP_MAX_NEW_TOKENS --max-new-tokens
forward MOTIONDROP_MAX_CONTEXT_TOKENS --max-context-tokens
forward MOTIONDROP_POSITION_MODE --position-mode
forward MOTIONDROP_STANDING_INTERVAL --standing-interval
forward MOTIONDROP_WATCH_INTERVAL --watch-interval
forward MOTIONDROP_MEMORY_INTERVAL --memory-interval
forward MOTIONDROP_MAX_GPU_MEMORY_GIB --max-gpu-memory-gib

cd "${ROOT_DIR}"
exec "${CLI}" demo ${ARGS[@]+"${ARGS[@]}"} "$@"
