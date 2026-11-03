#!/usr/bin/env bash
# Run StreamingBench. Defaults are those of `motiondrop streamingbench --help`;
# override them with MOTIONDROP_* variables (see scripts/lib.sh) or CLI options.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
require_cli

forward_benchmark_options
forward MOTIONDROP_STREAMING_DATA_ROOT --data-root
forward_list MOTIONDROP_STREAMING_DATASETS --datasets
forward MOTIONDROP_QUESTION_ID_FILE --question-id-file

cd "${ROOT_DIR}"
exec "${CLI}" streamingbench ${ARGS[@]+"${ARGS[@]}"} "$@"
