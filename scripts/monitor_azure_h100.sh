#!/usr/bin/env bash

set -euo pipefail

SUBSCRIPTION="${AZURE_SUBSCRIPTION_ID:-66679423-9d1a-4f45-8ae3-3078b8b62e99}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-ai}"
VM_NAME="${AZURE_VM_NAME:-vm-test-100}"
LOG_FILE="${1:-}"
WATCH="${WATCH:-false}"
INTERVAL="${INTERVAL:-30}"

if [[ "${LOG_FILE}" == "--watch" ]]; then
    LOG_FILE=""
    WATCH=true
fi

if [[ -n "${LOG_FILE}" && ! "${LOG_FILE}" =~ ^/mnt/kd-results/[A-Za-z0-9_./-]+\.log$ ]]; then
    echo "Invalid remote log path: ${LOG_FILE}" >&2
    exit 2
fi

snapshot() {
    local remote_script
    remote_script="$(cat <<EOF
set -eu
log_file='${LOG_FILE}'
if [ -z "\${log_file}" ]; then
  log_file=\$(find /mnt/kd-results -type f -name '*.log' -printf '%T@ %p\\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
fi
echo "LOG=\${log_file:-none}"
if [ -n "\${log_file}" ]; then
  pid_file="\${log_file%.log}.pid"
  if [ -f "\${pid_file}" ]; then
    pid=\$(tr -d '"' < "\${pid_file}")
    if kill -0 "\${pid}" 2>/dev/null; then
      echo "STATUS=running PID=\${pid}"
    else
      echo "STATUS=finished PID=\${pid}"
    fi
  elif grep -q '파이프라인 완료' "\${log_file}" 2>/dev/null; then
    echo 'STATUS=finished'
  elif pgrep -f '/mnt/kd-venv/bin/python main.py' >/dev/null 2>&1; then
    echo 'STATUS=running'
  else
    echo 'STATUS=starting-or-finished'
  fi
fi
echo 'GPU:'
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
echo 'PROGRESS:'
if [ -n "\${log_file}" ] && [ -f "\${log_file}" ]; then
  python3 -c 'import sys; from pathlib import Path; text = Path(sys.argv[1]).read_bytes()[-65536:].decode("utf-8", errors="replace"); print("\\n".join(text.splitlines()[-12:]))' "\${log_file}"
fi
EOF
)"

    az vm run-command invoke \
        --subscription "${SUBSCRIPTION}" \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${VM_NAME}" \
        --command-id RunShellScript \
        --scripts "${remote_script}" \
        --query 'value[0].message' \
        --output tsv
}

while true; do
    clear 2>/dev/null || true
    date
    snapshot
    if [[ "${WATCH}" != "true" ]]; then
        break
    fi
    sleep "${INTERVAL}"
done