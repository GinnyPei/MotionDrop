#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DEV=0

usage() {
  cat <<'EOF'
Usage: scripts/setup.sh [--dev]

Create the isolated MotionDrop environment and install Python dependencies.
This script does not download model checkpoints or benchmark datasets.

Examples:
  ./scripts/setup.sh
  ./scripts/setup.sh --dev

Pass --dev to also install the linter. Use download_models.sh
and download_datasets.sh after setup to fetch only the assets you need.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev)
      INSTALL_DEV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${INSTALL_DEV}" -eq 1 ]]; then
  exec "${ROOT_DIR}/scripts/install.sh" --dev
else
  exec "${ROOT_DIR}/scripts/install.sh"
fi
