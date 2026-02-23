#!/usr/bin/env bash
set -euo pipefail

PORT=7860
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

get_port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -iTCP:"${PORT}" -sTCP:LISTEN 2>/dev/null | sort -u || true
    return
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -ltnp "sport = :${PORT}" 2>/dev/null \
      | awk -F'pid=' 'NR>1 {split($2,a,","); if (a[1] ~ /^[0-9]+$/) print a[1]}' \
      | sort -u || true
    return
  fi

  return 0
}

terminate_pids() {
  local pids=("$@")
  if [ ${#pids[@]} -eq 0 ]; then
    return 0
  fi

  for pid in "${pids[@]}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done

  for _ in {1..20}; do
    local alive=0
    for pid in "${pids[@]}"; do
      if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
        alive=1
        break
      fi
    done
    [ "${alive}" -eq 0 ] && return 0
    sleep 0.2
  done

  for pid in "${pids[@]}"; do
    if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

clear_gpu_processes() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  mapfile -t gpu_pids < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF {print $1}' | sort -u)

  if [ ${#gpu_pids[@]} -gt 0 ]; then
    terminate_pids "${gpu_pids[@]}"
  fi
}

clear_gpu_processes

mapfile -t port_pids < <(get_port_pids)
if [ ${#port_pids[@]} -gt 0 ]; then
  terminate_pids "${port_pids[@]}"
fi

if command -v lsof >/dev/null 2>&1; then
  if lsof -t -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "端口 ${PORT} 仍被占用，启动失败" >&2
    exit 1
  fi
elif command -v ss >/dev/null 2>&1; then
  if ss -ltn "sport = :${PORT}" | awk 'NR>1 {exit 0} END {exit 1}'; then
    echo "端口 ${PORT} 仍被占用，启动失败" >&2
    exit 1
  fi
fi

sleep 2

cd "${ROOT_DIR}"

if [ "$#" -eq 0 ]; then
  exec "${PYTHON_BIN}" run_gradio.py --model AudioX --space-like-ui
else
  exec "${PYTHON_BIN}" run_gradio.py "$@"
fi
