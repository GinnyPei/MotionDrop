# Shared helpers for the scripts/run_*.sh wrappers (sourced, not executed).
#
# Wrappers never restate a default: the Python CLI owns every default, and a
# wrapper forwards a MOTIONDROP_* variable only when it is set.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MOTIONDROP_VENV:-${ROOT_DIR}/.venv}"
CLI="${VENV_DIR}/bin/motiondrop"
ARGS=()

require_cli() {
  if [[ ! -x "${CLI}" ]]; then
    echo "MotionDrop environment not found. Run: ${ROOT_DIR}/scripts/setup.sh" >&2
    exit 2
  fi
}

# forward VAR --flag: pass "--flag $VAR" when VAR is set and non-empty.
forward() {
  local value="${!1:-}"
  if [[ -n "${value}" ]]; then
    ARGS+=("$2" "${value}")
  fi
}

# forward_list VAR --flag: like forward, splitting a space-separated value.
forward_list() {
  local value="${!1:-}" items
  if [[ -n "${value}" ]]; then
    read -r -a items <<< "${value}"
    ARGS+=("$2" "${items[@]}")
  fi
}

# forward_switch VAR --flag: pass "--flag" or "--no-flag" when VAR is 1/0.
forward_switch() {
  case "${!1:-}" in
    1|true) ARGS+=("$2") ;;
    0|false) ARGS+=("--no-${2#--}") ;;
  esac
}

# Options shared by the OVO-Bench and StreamingBench runners.
forward_benchmark_options() {
  forward MOTIONDROP_MODEL --model
  forward MOTIONDROP_MODEL_PATH --model-path
  forward MOTIONDROP_OUTPUT --output
  forward MOTIONDROP_COMPRESSION_RATES --compression-rates
  forward MOTIONDROP_BATCH_SIZE --batch-size
  forward MOTIONDROP_MAX_FRAMES --max-frames
  forward MOTIONDROP_MIN_PIXELS --min-pixels
  forward MOTIONDROP_MAX_PIXELS --max-pixels
  forward MOTIONDROP_MAX_GPU_MEMORY_GIB --max-gpu-memory-gib
  forward MOTIONDROP_SEED --seed
  forward_motiondrop_options
}

# MotionDrop hyperparameters (see `motiondrop ovobench --help`).
forward_motiondrop_options() {
  forward MOTIONDROP_MOTION_NOISE_THRESHOLD --motion-noise-threshold
  forward MOTIONDROP_MOTION_LOCAL_RADIUS --motion-local-radius
  forward MOTIONDROP_FIRST_FRAME_POLICY --first-frame-policy
  forward MOTIONDROP_BUDGET_FLOOR --budget-floor
  forward MOTIONDROP_MOTION_POWER --motion-power
  forward_switch MOTIONDROP_DEDUP --dedup
}
