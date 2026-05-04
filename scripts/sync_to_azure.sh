#!/usr/bin/env bash
# 학습 서버에서 results/{logs,checkpoints,figures}/{run_id}/ 를 Azure Blob 으로 sync.
#
# 사전:
#   az login
#   export AZURE_STORAGE_ACCOUNT=<...>
#   export AZURE_BLOB_CONTAINER=kd-results
#
# 사용:
#   ./scripts/sync_to_azure.sh                # 가장 최근 run 1 개
#   ./scripts/sync_to_azure.sh 20260512_HHMMSS # 특정 run
#   ./scripts/sync_to_azure.sh --all          # results/ 전체

set -euo pipefail

ACCOUNT="${AZURE_STORAGE_ACCOUNT:?AZURE_STORAGE_ACCOUNT 환경변수 필요}"
CONTAINER="${AZURE_BLOB_CONTAINER:-kd-results}"
RESULTS_DIR="${RESULTS_DIR:-results}"

upload_run() {
  local run_id="$1"
  echo "==> sync run_id=${run_id}"

  for sub in logs checkpoints figures qualitative; do
    src="${RESULTS_DIR}/${sub}/${run_id}"
    if [[ ! -d "${src}" ]]; then
      continue
    fi
    echo "  - ${sub}/"
    az storage blob upload-batch \
      --auth-mode login \
      --account-name "${ACCOUNT}" \
      --destination "${CONTAINER}" \
      --destination-path "${sub}/${run_id}" \
      --source "${src}" \
      --overwrite \
      --no-progress 1>/dev/null
  done
  echo "    완료."
}

if [[ "${1:-}" == "--all" ]]; then
  for d in "${RESULTS_DIR}/logs"/*/; do
    rid="$(basename "${d}")"
    upload_run "${rid}"
  done
elif [[ -n "${1:-}" ]]; then
  upload_run "$1"
else
  latest="$(ls -t "${RESULTS_DIR}/logs" | head -1)"
  if [[ -z "${latest}" ]]; then
    echo "results/logs/ 에 run 이 없습니다." >&2
    exit 1
  fi
  upload_run "${latest}"
fi
