#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MOTIONDROP_VENV:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_DEV=0

usage() {
  cat <<'EOF'
Usage: scripts/install.sh [--dev]

Creates .venv, installs MotionDrop and its tested runtime dependencies, then
runs an environment check. Environment overrides:

  MOTIONDROP_VENV             virtual-environment path (default: .venv)
  PYTHON_BIN                  Python executable (default: python3)
  MOTIONDROP_TORCH_INDEX_URL  optional PyTorch wheel index URL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) INSTALL_DEV=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

"${PYTHON_BIN}" - <<'PY'
import sys
if not (3, 10) <= sys.version_info < (3, 13):
    raise SystemExit("MotionDrop requires Python 3.10, 3.11, or 3.12")
print(f"[install] Python {sys.version.split()[0]}")
PY

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[install] Creating virtual environment: ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[install] Reusing virtual environment: ${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel

if [[ -n "${MOTIONDROP_TORCH_INDEX_URL:-}" ]]; then
  echo "[install] Installing PyTorch and TorchVision from ${MOTIONDROP_TORCH_INDEX_URL}"
  "${VENV_DIR}/bin/python" -m pip install \
    --index-url "${MOTIONDROP_TORCH_INDEX_URL}" \
    "torch>=2.6,<3" "torchvision>=0.21,<1"
fi

if [[ "${INSTALL_DEV}" -eq 1 ]]; then
  "${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}[dev]"
else
  "${VENV_DIR}/bin/python" -m pip install -e "${ROOT_DIR}"
fi

echo "[install] Verifying installation"
"${VENV_DIR}/bin/motiondrop" doctor
echo "[install] Done. Activate with: source ${VENV_DIR}/bin/activate"
