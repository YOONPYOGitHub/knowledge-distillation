# UI 배포 가이드 — Azure Container Apps

로컬에서 개발한 UI(FastAPI 백엔드 + Streamlit 프론트엔드)를 Azure 에 그대로 올려서
팀원과 공유하기 위한 가이드. **학습은 별도 GPU 서버**에서 진행하고 결과만 Blob 으로
sync 하는 운영 모델을 가정함.

## 전체 구성

```
[학습 서버]                    [Azure Storage]              [Azure Container Apps]
  results/{logs,                 Blob: kd-results              ┌── frontend (Streamlit)
   checkpoints,    sync ───►       logs/{run_id}/...           │     팀원이 접속 (Entra ID 인증)
   figures}/{run_id}/             checkpoints/{run_id}/...     │
                                  figures/{run_id}/...    ◄── │── backend  (FastAPI)
                                  qualitative/{run_id}/...           Managed Identity 로 Blob 접근
                                                                     ckpt 는 lazy 다운로드 + 디스크 캐시
```

저장소(스토리지) 추상화는 환경변수 `STORAGE_BACKEND` 로 전환:

| 값 | 설명 |
|---|---|
| `local` | (기본) `results/` 폴더에서 직접 read/write |
| `blob`  | Azure Blob Storage 사용. `AZURE_STORAGE_ACCOUNT`, `AZURE_BLOB_CONTAINER` 필요 |

코드 변경 없이 동일한 UI 가 두 모드 모두에서 동작.

## 1. 사전 준비

```bash
az login
az account set --subscription "<구독>"
az group create -n rg-kd -l koreacentral
```

## 2. 인프라 1차 프로비저닝 (스토리지/ACR/환경)

이미지가 아직 없으니 `backendImage`/`frontendImage` 를 비워둔 채 먼저 띄움.

```bash
az deployment group create \
  -g rg-kd \
  -f infra/main.bicep \
  -p namePrefix=kdui location=koreacentral
```

출력에서 `acrLoginServer`, `storageAccountName` 을 확인.

## 3. 컨테이너 이미지 빌드 & ACR 푸시

```bash
ACR_NAME=<acrLoginServer 의 앞부분>     # 예: kduiacrxxxxx
./scripts/build_and_push.sh "${ACR_NAME}" v1
```

스크립트가 출력하는 두 줄을 메모.

```
backendImage  = <login>.azurecr.io/kd-backend:v1
frontendImage = <login>.azurecr.io/kd-frontend:v1
```

## 4. Container Apps 배포 (이미지 적용)

```bash
az deployment group create \
  -g rg-kd \
  -f infra/main.bicep \
  -p namePrefix=kdui location=koreacentral \
     backendImage='<login>.azurecr.io/kd-backend:v1' \
     frontendImage='<login>.azurecr.io/kd-frontend:v1'
```

출력의 `frontendUrl` 이 팀원이 접속할 주소.

## 5. 학습 결과 sync (학습 서버에서)

```bash
export AZURE_STORAGE_ACCOUNT=<storageAccountName>
export AZURE_BLOB_CONTAINER=kd-results
az login

# 가장 최근 run 1개만
./scripts/sync_to_azure.sh

# 특정 run
./scripts/sync_to_azure.sh 20260512_HHMMSS

# 전체 (초기 1회)
./scripts/sync_to_azure.sh --all
```

업로드 후 UI 새로고침하면 드롭다운에 새 run 이 자동 노출됨.

## 6. (선택) 팀 인증 추가

Container Apps 의 frontend 앱에 Entra ID 인증을 켜면 회사 계정 사용자만 접속 가능:

```bash
# 포털 GUI 로 켜는 게 가장 간단
# Container App → Authentication → Add identity provider → Microsoft
```

## 운영 팁

- **비용 절감**: 기본 `minReplicas=0` 이라 사용 없는 시간엔 0원. 첫 접속 시 콜드 스타트 수 초.
- **체크포인트 캐시**: 백엔드 컨테이너 `/tmp/kd-cache/` 에 LRU 로 4GB 까지 저장.
  Container Apps 가 0 으로 스케일 다운되면 캐시가 사라짐 → 다음 첫 추론 요청에서 재다운로드.
- **새 run 알림**: `sync_to_azure.sh` 끝에 Slack/Teams webhook 한 줄 추가 권장.
- **로그 확인**:
  ```bash
  az containerapp logs show -g rg-kd -n kdui-backend  --follow
  az containerapp logs show -g rg-kd -n kdui-frontend --follow
  ```

## 환경변수 요약 (백엔드)

| 변수 | 기본 | 설명 |
|---|---|---|
| `STORAGE_BACKEND` | `local` | `local` 또는 `blob` |
| `AZURE_STORAGE_ACCOUNT` | — | Blob 모드일 때 필수 |
| `AZURE_BLOB_CONTAINER` | `kd-results` | Blob 컨테이너 이름 |
| `AZURE_BLOB_CACHE_DIR` | `/tmp/kd-cache` | ckpt 다운로드 캐시 위치 |
| `AZURE_BLOB_CACHE_MAX_GB` | `4` | 캐시 최대 용량 (GB) |
| `MODEL_CACHE_SIZE` | `2` | 동시에 메모리에 둘 run 수 |
| `DEVICE` | `auto` | `cpu` / `cuda` / `mps` / `auto` |
| `DEFAULT_RUN_ID` | (비움) | 부팅 시 사전 로드할 run |
