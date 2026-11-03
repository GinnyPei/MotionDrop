#!/usr/bin/env bash
# Run OVO-Bench. Defaults are those of `motiondrop ovobench --help`; override
# them with MOTIONDROP_* variables (see scripts/lib.sh) or extra CLI options.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
require_cli

forward_benchmark_options
forward MOTIONDROP_OVO_ANNOTATIONS --annotations
forward MOTIONDROP_OVO_VIDEOS --videos
forward_list MOTIONDROP_OVO_TASKS --tasks
forward MOTIONDROP_OVO_FPS --fps

cd "${ROOT_DIR}"
exec "${CLI}" ovobench ${ARGS[@]+"${ARGS[@]}"} "$@"
