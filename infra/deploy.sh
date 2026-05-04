#!/usr/bin/env bash
# 두 단계 배포 헬퍼.
#
#   ./infra/deploy.sh prereqs               # Storage/ACR/Env/UAMI/권한
#   ./infra/deploy.sh apps <tag>            # Container Apps (이미지 태그 지정)
#   ./infra/deploy.sh all <tag>             # prereqs + 빌드 + apps
#
# 환경변수 (옵션):
#   RG=rg-kd
#   LOCATION=koreacentral
#   NAME_PREFIX=kdui

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "사용: $0 prereqs|apps|all [tag]" >&2
  exit 1
fi
MODE="$1"
TAG="${2:-v1}"

RG="${RG:-rg-kd}"
LOCATION="${LOCATION:-koreacentral}"
NAME_PREFIX="${NAME_PREFIX:-kdui}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BICEP="${PROJECT_ROOT}/infra/main.bicep"

deploy_prereqs() {
  echo "==> prereqs 배포 (RG=${RG})"
  az group create -n "${RG}" -l "${LOCATION}" -o none

  az deployment group create \
    -g "${RG}" \
    -f "${BICEP}" \
    -p namePrefix="${NAME_PREFIX}" location="${LOCATION}" deployMode=prereqs \
    -o none

  ACR_NAME="$(az deployment group show -g "${RG}" -n main --query properties.outputs.acrName.value -o tsv)"
  echo "    ACR_NAME=${ACR_NAME}"
  echo "${ACR_NAME}"
}

deploy_apps() {
  ACR_NAME="$(az acr list -g "${RG}" --query "[?starts_with(name, '${NAME_PREFIX}acr')] | [0].name" -o tsv 2>/dev/null || true)"
  if [[ -z "${ACR_NAME}" ]]; then
    echo "ACR 을 찾을 수 없음 (RG=${RG}, prefix=${NAME_PREFIX}acr). 먼저 prereqs 배포 필요." >&2
    exit 1
  fi
  LOGIN_SERVER="$(az acr show -n "${ACR_NAME}" --query loginServer -o tsv)"

  echo "==> apps 배포 (tag=${TAG})"
  az deployment group create \
    -g "${RG}" \
    -f "${BICEP}" \
    -p namePrefix="${NAME_PREFIX}" location="${LOCATION}" deployMode=apps \
       backendImage="${LOGIN_SERVER}/kd-backend:${TAG}" \
       frontendImage="${LOGIN_SERVER}/kd-frontend:${TAG}" \
    --query "properties.outputs.{frontendUrl:frontendUrl, backendUrl:backendUrl}" \
    -o json
}

case "${MODE}" in
  prereqs)
    deploy_prereqs
    ;;
  apps)
    deploy_apps
    ;;
  all)
    ACR_NAME="$(deploy_prereqs | tail -1)"
    "${PROJECT_ROOT}/scripts/build_and_push.sh" "${ACR_NAME}" "${TAG}"
    deploy_apps
    ;;
  *)
    echo "알 수 없는 모드: ${MODE}" >&2
    exit 1
    ;;
esac
