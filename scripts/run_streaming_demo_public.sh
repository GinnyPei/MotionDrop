#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_HOST="${MOTIONDROP_DEMO_BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${MOTIONDROP_DEMO_BACKEND_PORT:-7860}"
PUBLIC_HOST="${MOTIONDROP_DEMO_PUBLIC_HOST:-0.0.0.0}"
PUBLIC_PORT="${MOTIONDROP_DEMO_PUBLIC_PORT:-80}"
STARTUP_TIMEOUT="${MOTIONDROP_DEMO_STARTUP_TIMEOUT:-300}"
PROXY_PYTHON="${MOTIONDROP_PROXY_PYTHON:-/usr/bin/python3}"

if [[ ! -x "${PROXY_PYTHON}" ]]; then
  echo "Python interpreter not found: ${PROXY_PYTHON}" >&2
  exit 2
fi
if [[ ! "${BACKEND_PORT}" =~ ^[0-9]+$ || ! "${PUBLIC_PORT}" =~ ^[0-9]+$ ]]; then
  echo "Demo ports must be integers." >&2
  exit 2
fi
if (( BACKEND_PORT < 1 || BACKEND_PORT > 65535 || PUBLIC_PORT < 1 || PUBLIC_PORT > 65535 )); then
  echo "Demo ports must be between 1 and 65535." >&2
  exit 2
fi

backend_pid=""
proxy_pid=""
proxy_uses_sudo=0

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${proxy_pid}" ]] && kill -0 "${proxy_pid}" 2>/dev/null; then
    if (( proxy_uses_sudo )); then
      sudo -n kill "${proxy_pid}" 2>/dev/null || true
    else
      kill "${proxy_pid}" 2>/dev/null || true
    fi
  fi
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    kill "${backend_pid}" 2>/dev/null || true
    wait "${backend_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if (( PUBLIC_PORT < 1024 )); then
  echo "Port ${PUBLIC_PORT} requires administrator permission; sudo will prompt once."
  sudo -v
  proxy_uses_sudo=1
fi

MOTIONDROP_DEMO_HOST="${BACKEND_HOST}" \
MOTIONDROP_DEMO_PORT="${BACKEND_PORT}" \
  "${ROOT_DIR}/scripts/run_streaming_demo.sh" "$@" &
backend_pid=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until nc -z "${BACKEND_HOST}" "${BACKEND_PORT}" 2>/dev/null; do
  if ! kill -0 "${backend_pid}" 2>/dev/null; then
    wait "${backend_pid}"
    exit $?
  fi
  if (( SECONDS >= deadline )); then
    echo "Demo backend did not become ready within ${STARTUP_TIMEOUT} seconds." >&2
    exit 1
  fi
  sleep 1
done

proxy_args=(
  "${ROOT_DIR}/scripts/http_port_proxy.py"
  --listen-host "${PUBLIC_HOST}"
  --listen-port "${PUBLIC_PORT}"
  --target-host "${BACKEND_HOST}"
  --target-port "${BACKEND_PORT}"
)
if (( proxy_uses_sudo )); then
  sudo -n "${PROXY_PYTHON}" "${proxy_args[@]}" &
else
  "${PROXY_PYTHON}" "${proxy_args[@]}" &
fi
proxy_pid=$!

sleep 0.3
if ! kill -0 "${proxy_pid}" 2>/dev/null; then
  wait "${proxy_pid}"
  exit $?
fi

server_ip="$(ip -4 route show default 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i == "src") {print $(i + 1); exit}}')"
if [[ -z "${server_ip}" ]]; then
  server_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
if [[ -n "${server_ip}" ]]; then
  if [[ "${PUBLIC_PORT}" == "80" ]]; then
    echo "MotionDrop demo: http://${server_ip}"
  else
    echo "MotionDrop demo: http://${server_ip}:${PUBLIC_PORT}"
  fi
else
  echo "MotionDrop demo is listening on ${PUBLIC_HOST}:${PUBLIC_PORT}."
fi
echo "Press Ctrl+C to stop the demo."

wait "${proxy_pid}"
