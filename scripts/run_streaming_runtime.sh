#!/usr/bin/env bash
# Run the pipelined streaming runtime on a sample OVO-Bench video.
#
# By default the window budget comes from the fitted latency profile and an
# SLO; set MOTIONDROP_FIXED_COMPRESSION to use a fixed ratio instead. Other
# defaults are those of `motiondrop stream --help`.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
require_cli

ARGS=(
  --query "${MOTIONDROP_STREAM_QUERY:-Describe what happened in the video.}"
  --max-stream-frames "${MOTIONDROP_MAX_STREAM_FRAMES:-8}"
)
if [[ -n "${MOTIONDROP_FIXED_COMPRESSION:-}" ]]; then
  ARGS+=(--fixed-compression "${MOTIONDROP_FIXED_COMPRESSION}")
else
  ARGS+=(--latency-model "${MOTIONDROP_LATENCY_MODEL:-${ROOT_DIR}/streaming_runtime/profiles/llava-ov-7b.json}")
  forward MOTIONDROP_SLO_MS --slo-ms
fi
forward MOTIONDROP_STREAM_VIDEO --video
forward MOTIONDROP_MODEL_PATH --model-path
forward MOTIONDROP_OUTPUT --output
forward MOTIONDROP_STREAM_FPS --fps
forward MOTIONDROP_QUERY_AT_SECONDS --query-at-seconds
forward MOTIONDROP_WINDOW_FRAMES --window-frames
forward MOTIONDROP_QUEUE_SIZE --queue-size
forward MOTIONDROP_MAX_NEW_TOKENS --max-new-tokens
forward MOTIONDROP_MAX_GPU_MEMORY_GIB --max-gpu-memory-gib
forward_motiondrop_options

cd "${ROOT_DIR}"
exec "${CLI}" stream "${ARGS[@]}" "$@"
