# UI 배포 가이드 — Azure Container Apps

로컬에서 개발한 UI(FastAPI 백엔드 + Streamlit 프론트엔드)를 Azure 에 그대로 올려서
팀원과 공유하기 위한 가이드. **학습은 별도 GPU 서버**에서 진행하고 결과만 Blob 으로
sync 하는 운영 모델을 가정함.

## 전체 구성

```
[학습 서버]                    [Azure Blob (PE)]            [Azure Container Apps (VNet)]
  results/{logs,                 kd-results 컨테이너            ┌── frontend (Streamlit, external)
   checkpoints,    sync ───►       logs/{run_id}/...           │     팀원이 https URL 로 접속
   figures,                        checkpoints/{run_id}/...    │
   qualitative}/{run_id}/          figures/{run_id}/...    ◄── │── backend  (FastAPI, internal-only)
                                   qualitative/{run_id}/...          UAMI → Blob (Private Endpoint)
                                                                     ckpt lazy download + LRU 디스크 캐시
```

- Storage 는 **Private Endpoint** 로만 접근 (public 차단)
- Container Apps Environment 가 VNet 에 통합되어 있어서 backend 가 PE 통해 Blob 접근
- Frontend → Backend 호출은 ManagedEnv 내부 통신
- 외부 사용자는 Frontend 의 외부 https URL 로 접속

저장소 추상화는 환경변수 `STORAGE_BACKEND` 로 전환:

| 값 | 설명 |
|---|---|
| `local` | (기본) `results/` 폴더에서 직접 read/write |
| `blob`  | Azure Blob Storage 사용. `AZURE_STORAGE_ACCOUNT`, `AZURE_BLOB_CONTAINER` 필요 |

## 0. 사전 요구사항

| 항목 | 필요 권한/도구 |
|---|---|
| Azure CLI 로그인 | `az login` |
| 구독 권한 | Owner 또는 Contributor + User Access Administrator (role assignment 생성용) |
| Bicep CLI | `az` 가 자동 처리 |
| 도커 데몬 | **불필요** — `az acr build` 로 클라우드 빌드 |

## 1. 한 번에 배포 (권장)

```bash
az login
az account set --subscription "<구독>"

./infra/deploy.sh all v1
```

이 한 줄이 다음을 모두 수행:

1. **네트워크 사전 리소스** (멱등; 이미 있으면 스킵)
   - `rg-mgmt` RG, `VNET-KRC` (10.1.0.0/16)
   - `snet-aca-infra` (10.1.4.0/23, ACA 위임)
   - `snet-pe` (10.1.6.0/27, PE 정책 disabled)
   - `privatelink.blob.core.windows.net` Private DNS Zone + VNet Link
2. **rg-kd** prereqs (Bicep)
   - Storage (`publicNetworkAccess=Disabled`, `allowSharedKeyAccess=false`)
   - Storage Private Endpoint + DNS A record
   - ACR / Log Analytics / UAMI / VNet 통합 ManagedEnv
   - UAMI 에 AcrPull / Storage Blob Data Contributor 부여
3. **이미지 빌드 & ACR push** (`./scripts/build_and_push.sh`)
   - `kd-backend:v1`, `kd-frontend:v1`
4. **Container Apps** 배포 (Bicep)
   - backend (internal, 2 vCPU / 4 GiB)
   - frontend (external, 0.5 vCPU / 1 GiB)
   - 둘 다 UAMI 사용

배포 끝에 `frontendUrl` 이 출력되며, 그 URL 이 팀원이 접속할 주소.

### 환경변수로 오버라이드

```bash
RG=rg-kd \
LOCATION=koreacentral \
NAME_PREFIX=kdui \
VNET_RG=rg-mgmt \
VNET_NAME=VNET-KRC \
ACA_SUBNET_CIDR=10.1.4.0/23 \
PE_SUBNET_CIDR=10.1.6.0/27 \
./infra/deploy.sh all v1
```

## 2. 단계별 배포 (선택)

문제 진단이나 부분 갱신이 필요하면 단계 분리.

### 2-1. prereqs 만 (인프라 + 권한)

```bash
./infra/deploy.sh prereqs
```

### 2-2. 이미지 빌드 & push 만

```bash
ACR_NAME="$(az acr list -g rg-kd --query "[0].name" -o tsv)"
./scripts/build_and_push.sh "${ACR_NAME}" v2
```

### 2-3. apps 만 (이미지 새 태그 적용)

```bash
./infra/deploy.sh apps v2
```

## 3. 학습 결과 sync (학습 서버에서)

```bash
az login
export AZURE_STORAGE_ACCOUNT=<storageAccountName>
export AZURE_BLOB_CONTAINER=kd-results

# 가장 최근 run 1개만
./scripts/sync_to_azure.sh

# 특정 run
./scripts/sync_to_azure.sh 20260512_HHMMSS

# 전체 (초기 1회)
./scripts/sync_to_azure.sh --all
```

> Storage public access 가 차단되어 있으므로, 학습 서버는 다음 중 하나여야 함:
> - 같은 VNet (또는 peered VNet) 내부
> - 동일 구독의 Azure 리소스 (`bypass: AzureServices` 로 통과)
> - 또는 임시로 `publicNetworkAccess=Enabled` + IP allowlist 로 허용 후 다시 차단

## 4. (선택) 팀 인증

Container Apps 의 frontend 앱에 Entra ID 인증을 켜면 회사 계정 사용자만 접속 가능:

```
포털 GUI: Container App → Authentication → Add identity provider → Microsoft
```

## 5. 운영 팁

- **비용**: `minReplicas=0` 이라 사용 없는 시간엔 active 요금 0원. 콜드 스타트 수 초.
- **체크포인트 캐시**: backend 컨테이너 `/tmp/kd-cache/` 에 LRU 로 4 GiB 까지.
  Replica 가 0 으로 내려가면 캐시 사라짐 → 첫 추론에서 재다운로드.
- **figure PNG**: backend 가 PE 환경에서 SAS redirect 대신 inline bytes 로 응답.
  Frontend Streamlit 도 server-side fetch 후 `st.image(bytes)` 로 렌더.
- **로그 확인**:
  ```bash
  az containerapp logs show -g rg-kd -n kdui-backend  --tail 100
  az containerapp logs show -g rg-kd -n kdui-frontend --tail 100
  ```
- **이미지 롤백**:
  ```bash
  az containerapp update -g rg-kd -n kdui-backend --image <acr>.azurecr.io/kd-backend:v1
  ```
- **임시 정지**:
  ```bash
  az containerapp update -g rg-kd -n kdui-backend  --min-replicas 0 --max-replicas 0
  az containerapp update -g rg-kd -n kdui-frontend --min-replicas 0 --max-replicas 0
  ```

## 6. 환경변수 요약 (backend)

| 변수 | 기본 | 설명 |
|---|---|---|
| `STORAGE_BACKEND` | `local` | `local` 또는 `blob` |
| `AZURE_STORAGE_ACCOUNT` | — | Blob 모드일 때 필수 |
| `AZURE_BLOB_CONTAINER` | `kd-results` | Blob 컨테이너 이름 |
| `AZURE_CLIENT_ID` | — | UAMI client ID (DefaultAzureCredential 힌트) |
| `AZURE_BLOB_CACHE_DIR` | `/tmp/kd-cache` | ckpt 다운로드 캐시 위치 |
| `AZURE_BLOB_CACHE_MAX_GB` | `4` | 캐시 최대 용량 (GB) |
| `MODEL_CACHE_SIZE` | `2` | 동시에 메모리에 둘 run 수 |
| `DEVICE` | `auto` | `cpu` / `cuda` / `mps` / `auto` |
| `DEFAULT_RUN_ID` | (비움) | 부팅 시 사전 로드할 run |

## 7. 비용 (Korea Central, 참고)

| 시나리오 | 월 예상 |
|---|---|
| `minReplicas=0`, 가끔 사용 (데모/리뷰) | $15–25 |
| 평일 9–18시 상시 가동 (min=1) | $85–95 |
| 24/7 상시 (min=1) | $285–295 |

고정비: ACR Basic ~$5, Private Endpoint ~$7.3/월.
