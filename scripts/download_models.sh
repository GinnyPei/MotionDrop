#!/usr/bin/env bash
# Download checkpoints/datasets; see `motiondrop download --help` for options.
set -Eeuo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/lib.sh"
require_cli

cd "${ROOT_DIR}"
exec "${CLI}" download --datasets none "$@"
