#!/usr/bin/env bash
# 두 단계 배포 헬퍼.
#
#   ./infra/deploy.sh prereqs               # Network 사전 준비 + Storage/ACR/Env/UAMI/권한
#   ./infra/deploy.sh apps <tag>            # Container Apps (이미지 태그 지정)
#   ./infra/deploy.sh all <tag>             # prereqs + 빌드 + apps
#
# 환경변수 (옵션):
#   RG=rg-kd                  - 앱 리소스 RG
#   LOCATION=koreacentral
#   NAME_PREFIX=kdui
#   VNET_RG=rg-mgmt           - 공용 네트워크 RG (다른 워크로드와 공유 가능)
#   VNET_NAME=VNET-KRC
#   VNET_CIDR=10.1.0.0/16
#   ACA_SUBNET=snet-aca-infra
#   ACA_SUBNET_CIDR=10.1.4.0/23
#   PE_SUBNET=snet-pe
#   PE_SUBNET_CIDR=10.1.6.0/27
#   DNS_ZONE=privatelink.blob.core.windows.net

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

VNET_RG="${VNET_RG:-rg-mgmt}"
VNET_NAME="${VNET_NAME:-VNET-KRC}"
VNET_CIDR="${VNET_CIDR:-10.1.0.0/16}"
ACA_SUBNET="${ACA_SUBNET:-snet-aca-infra}"
ACA_SUBNET_CIDR="${ACA_SUBNET_CIDR:-10.1.4.0/23}"
PE_SUBNET="${PE_SUBNET:-snet-pe}"
PE_SUBNET_CIDR="${PE_SUBNET_CIDR:-10.1.6.0/27}"
DNS_ZONE="${DNS_ZONE:-privatelink.blob.core.windows.net}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BICEP="${PROJECT_ROOT}/infra/main.bicep"

# ---- 사전 네트워크 보장 (멱등) -------------------------------------------------
ensure_network() {
  echo "==> 네트워크 사전 준비 (VNET_RG=${VNET_RG}, VNET=${VNET_NAME})"

  # 1) VNet 용 RG
  if ! az group show -n "${VNET_RG}" -o none 2>/dev/null; then
    echo "  RG 생성: ${VNET_RG}"
    az group create -n "${VNET_RG}" -l "${LOCATION}" -o none
  else
    echo "  RG 존재: ${VNET_RG}"
  fi

  # 2) VNet
  if ! az network vnet show -g "${VNET_RG}" -n "${VNET_NAME}" -o none 2>/dev/null; then
    echo "  VNet 생성: ${VNET_NAME} (${VNET_CIDR})"
    az network vnet create -g "${VNET_RG}" -n "${VNET_NAME}" \
      --address-prefixes "${VNET_CIDR}" -l "${LOCATION}" -o none
  else
    echo "  VNet 존재: ${VNET_NAME}"
  fi

  # 3) Subnet: ACA infra (Microsoft.App/environments 위임 필요)
  if ! az network vnet subnet show -g "${VNET_RG}" --vnet-name "${VNET_NAME}" -n "${ACA_SUBNET}" -o none 2>/dev/null; then
    echo "  Subnet 생성: ${ACA_SUBNET} (${ACA_SUBNET_CIDR})"
    az network vnet subnet create -g "${VNET_RG}" --vnet-name "${VNET_NAME}" -n "${ACA_SUBNET}" \
      --address-prefixes "${ACA_SUBNET_CIDR}" \
      --delegations Microsoft.App/environments -o none
  else
    echo "  Subnet 존재: ${ACA_SUBNET}"
  fi

  # 4) Subnet: Private Endpoint
  if ! az network vnet subnet show -g "${VNET_RG}" --vnet-name "${VNET_NAME}" -n "${PE_SUBNET}" -o none 2>/dev/null; then
    echo "  Subnet 생성: ${PE_SUBNET} (${PE_SUBNET_CIDR})"
    az network vnet subnet create -g "${VNET_RG}" --vnet-name "${VNET_NAME}" -n "${PE_SUBNET}" \
      --address-prefixes "${PE_SUBNET_CIDR}" \
      --private-endpoint-network-policies Disabled -o none
  else
    echo "  Subnet 존재: ${PE_SUBNET}"
  fi

  # 5) Private DNS Zone
  if ! az network private-dns zone show -g "${VNET_RG}" -n "${DNS_ZONE}" -o none 2>/dev/null; then
    echo "  Private DNS Zone 생성: ${DNS_ZONE}"
    az network private-dns zone create -g "${VNET_RG}" -n "${DNS_ZONE}" -o none
  else
    echo "  Private DNS Zone 존재: ${DNS_ZONE}"
  fi

  # 6) DNS Zone ↔ VNet link (이름 아닌 VNet ID로 존재 확인)
  local vnet_id existing_link
  vnet_id="$(az network vnet show -g "${VNET_RG}" -n "${VNET_NAME}" --query id -o tsv)"
  existing_link="$(az network private-dns link vnet list -g "${VNET_RG}" -z "${DNS_ZONE}" \
    --query "[?virtualNetwork.id=='${vnet_id}'] | [0].name" -o tsv 2>/dev/null || true)"
  if [[ -z "${existing_link}" ]]; then
    local link_name="${VNET_NAME}-link"
    echo "  DNS Zone link 생성: ${link_name}"
    az network private-dns link vnet create -g "${VNET_RG}" -z "${DNS_ZONE}" -n "${link_name}" \
      --virtual-network "${vnet_id}" --registration-enabled false -o none
  else
    echo "  DNS Zone link 존재: ${existing_link}"
  fi
}

deploy_prereqs() {
  ensure_network

  echo "==> prereqs 배포 (RG=${RG})"
  az group create -n "${RG}" -l "${LOCATION}" -o none

  az deployment group create \
    -g "${RG}" \
    -f "${BICEP}" \
    -p namePrefix="${NAME_PREFIX}" location="${LOCATION}" deployMode=prereqs \
       vnetResourceGroup="${VNET_RG}" vnetName="${VNET_NAME}" \
       acaSubnetName="${ACA_SUBNET}" peSubnetName="${PE_SUBNET}" \
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
       vnetResourceGroup="${VNET_RG}" vnetName="${VNET_NAME}" \
       acaSubnetName="${ACA_SUBNET}" peSubnetName="${PE_SUBNET}" \
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
