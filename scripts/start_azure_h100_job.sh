#!/usr/bin/env bash

set -euo pipefail

SUBSCRIPTION="${AZURE_SUBSCRIPTION_ID:-66679423-9d1a-4f45-8ae3-3078b8b62e99}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-ai}"
VM_NAME="${AZURE_VM_NAME:-vm-test-100}"
CONFIG="${1:-configs/h100_qwen_korean.yaml}"
shift || true

RUNNER="${AZURE_H100_RUNNER:-scripts/run_azure_h100.sh}"

if [[ ! "${RUNNER}" =~ ^scripts/[A-Za-z0-9_-]+\.sh$ ]]; then
    echo "Invalid runner path: ${RUNNER}" >&2
    exit 2
fi

if [[ ! "${CONFIG}" =~ ^[A-Za-z0-9_./-]+\.yaml$ ]]; then
    echo "Invalid config path: ${CONFIG}" >&2
    exit 2
fi

for argument in "$@"; do
    if [[ ! "${argument}" =~ ^[A-Za-z0-9_./-]+$ ]]; then
        echo "Invalid argument: ${argument}" >&2
        exit 2
    fi
done

job_name="$(basename "${CONFIG}" .yaml)-$(date +%Y%m%d-%H%M%S)"
remote_log="/mnt/kd-results/jobs/${job_name}.log"
remote_pid="/mnt/kd-results/jobs/${job_name}.pid"
quoted_args=""
for argument in "$@"; do
    printf -v quoted_argument '%q' "${argument}"
    quoted_args+=" ${quoted_argument}"
done

remote_script="$(cat <<EOF
set -eu
mkdir -p /mnt/kd-results/jobs
cd /mnt/knowledge-distillation
nohup env HF_HOME=/mnt/huggingface HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false \\
    bash '${RUNNER}' '${CONFIG}'${quoted_args} \\
  > '${remote_log}' 2>&1 < /dev/null &
pid=\$!
echo "\${pid}" > '${remote_pid}'
echo 'JOB_NAME=${job_name}'
echo "PID=\${pid}"
echo 'LOG=${remote_log}'
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