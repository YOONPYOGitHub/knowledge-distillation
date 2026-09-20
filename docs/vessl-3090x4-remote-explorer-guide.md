# VESSL Remote Explorer: OS별 접속 안내와 3090 × 4 공통 실행 가이드

이 가이드는 특정 계정·조직·워크스페이스에 종속되지 않습니다. **접속만 필요하면 자신의 OS 안내를 선택**하고, 이 저장소의 KD 실험을 실행할 경우 접속 후 아래 공통 절차를 진행하세요.

| 로컬 컴퓨터 | 접속 매뉴얼 | 로컬 명령어 환경 |
|---|---|---|
| Windows 10/11 | [Windows 단계별 안내](vessl-remote-explorer-windows.md) | PowerShell |
| macOS | [macOS 단계별 안내](vessl-remote-explorer-macos.md) | 기본 Terminal / zsh |

각 안내는 준비 도구 → CLI 설치 → SSH 키 → 계정/조직 확인 → Remote Explorer 접속 → 원격 폴더 열기 → 재접속 순서입니다. **로컬 OS가 Windows여도 VESSL의 원격 OS는 Linux**입니다.

## 0. 공통 실행 절차를 시작하기 전에

- 아래 명령은 모두 **SSH로 연결된 VESSL 원격 Linux 터미널**에서 실행합니다. Windows PowerShell이나 Mac 로컬 터미널에 입력하지 마세요.
- 계정·조직·워크스페이스·Host 별칭은 각 사용자의 값을 사용합니다. GPU 종류/개수와 저장소 할당량도 직접 확인합니다.
- 이 저장소의 학습 안내는 `feature/vessl-3090x4-h100-parity` 브랜치와 **RTX 3090 24GB × 4**를 기준으로 합니다. SSH 접속 자체에는 GPU 4장이 필요하지 않습니다.
- 실행 기준: [4-GPU 설정](../configs/vessl_qwen7b_korean_3090x4.yaml), [런처](../scripts/run_vessl.sh). 작은 모델의 4-GPU 검증과 실제 7B 학습 검증을 구분합니다.
- 원본 체크포인트가 없으면 `check`·`smoke`까지만 진행합니다. `READY`는 파일 존재/장치 확인이며, 7B 메모리 적합성·adapter 호환성·속도 향상의 증명이 아닙니다.

## 1. [VESSL 원격] 작업 폴더와 브랜치 확인

**File → Open Folder…**에서 서버에 받은 프로젝트를 엽니다. 아래 경로는 예시이므로 실제 위치로 바꾸세요. 새 터미널을 열면 `PROJECT_DIR`도 다시 지정합니다.

```bash
PROJECT_DIR="$HOME/knowledge-distillation"
cd "$PROJECT_DIR"
hostname
pwd
git branch --show-current
git log -1 --oneline
git status --short
```

- `hostname`은 원격 컨테이너 이름이며 SSH 별칭과 같을 필요는 없습니다. 로컬 PC 이름이면 잘못된 창입니다.
- `pwd`는 위 작업 사본, 브랜치는 `feature/vessl-3090x4-h100-parity`여야 합니다.
- 프로젝트가 없으면 브랜치가 게시된 저장소에서 clone하거나, 관리자가 소스와 Git 이력을 전달해야 합니다. **게시되지 않은 브랜치나 미커밋 변경은 일반 clone/pull로 전달되지 않습니다.**
- 파일만 전달받아 Git 정보가 없다면 관리자로부터 소스 버전을 확인합니다. 원본/백업 사본을 실행용 사본과 구분하고, 변경을 없애려고 `git reset --hard`/`git clean`을 실행하지 마세요.
- **Windows/macOS 가상환경·개인키·인증 파일은 서버로 복사하지 않습니다.**

## 2. [VESSL 원격] Python 환경과 GPU·디스크 점검

이미 만든 원격 가상환경을 우선 사용합니다. 아래 생성 명령은 가상환경이 없을 때만 동작합니다.

```bash
cd "$PROJECT_DIR"
python3 --version
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install -r requirements-vessl.txt
.venv/bin/python -m pip check
nvidia-smi
.venv/bin/python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print("visible GPUs:", torch.cuda.device_count())'
```

이 저장소에서 검증한 이미지 조합은 Python **3.10.12**, PyTorch **2.3.1+cu121**, 드라이버 **550.107.02**입니다. 사용자의 VESSL 이미지가 다르면 먼저 호환성을 확인하세요. 위 명령은 이미지의 CUDA PyTorch를 공유하며, CPU용 torch나 다른 CUDA 버전으로 임의 교체하지 않습니다.

[../requirements-vessl.txt](../requirements-vessl.txt)은 Transformers 4.45.2, tokenizers 0.20.3, PEFT 0.13.2, datasets 3.6.0, Accelerate 0.34.2, NumPy < 2 조합입니다. 오류/버전 불일치가 있으면 학습 전에 해결합니다.

VS Code 명령 팔레트(**Windows: Ctrl+Shift+P / macOS: Cmd+Shift+P**)에서 **Python: Select Interpreter → Enter interpreter path…**를 선택합니다. 다음 명령이 출력한 경로를 지정하세요. Python 명령이 없으면 Python 확장을 **SSH 대상에 설치**합니다.

```bash
printf '%s/.venv/bin/python\n' "$PWD"
```

GPU는 4개 모두 보여야 합니다. `nvidia-smi`의 4개와 Python의 `visible GPUs: 4`를 함께 확인합니다.

```bash
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<미설정>}"
df -h "$PROJECT_DIR" "$HOME"
du -sh "$PROJECT_DIR" "$HOME/.cache"
```

저장소 용량·보존 정책은 워크스페이스마다 다릅니다. `df`의 호스트 루트 용량과 실제 할당량이 다를 수 있으므로 **VESSL 화면의 저장소 사용량/할당량도 확인**합니다. 체크포인트·모델 캐시·새 실행 결과를 위한 여유가 필요합니다.

## 3. [VESSL 원격] 준비 상태 확인 → 작은 4GPU 검증

실험 실행은 반드시 이 런처만 사용합니다. 마지막 인자를 생략해도 **기본값은 `check`이며 학습하지 않습니다.**

```bash
bash scripts/run_vessl.sh configs/vessl_qwen7b_korean_3090x4.yaml check
```

원본 파일이 없으면 `BLOCKED: Missing original Azure …`와 `NOT READY: no training started`가 예상됩니다. **`check`는 미준비 상태에서도 종료 코드 0이므로 출력 내용을 확인**하세요. 모델 다운로드는 하지 않습니다.

GPU/환경에 문제가 없으면 체크포인트 없이도 다음 검증을 실행할 수 있습니다.

```bash
bash scripts/run_vessl.sh configs/vessl_qwen7b_korean_3090x4.yaml smoke
```

성공 시 기대 출력:

```text
rank=0 student=cuda:0 teacher=cuda:2 ... PASS
rank=1 student=cuda:1 teacher=cuda:3 ... PASS
PAIRED_4GPU_PARITY_PASSED (tiny models only; full-model fit/speed unverified)
```

작은 Qwen을 오프라인으로 생성해 **Student 2 ranks + 전용 Teacher 2개**를 검증합니다. FP32 forward/reverse KL 비교, BF16 실행, 불균등 유효 토큰/마지막 배치, rank 간 가중치 일치를 확인합니다. **BF16의 단일 rank 대비 비트 동일성, 실제 7B의 VRAM·속도·PPL은 보장하지 않습니다.** 실패하면 로그를 보존하고 여기서 중단합니다.

## 4. STOP: 비교 대상 실험의 원본 체크포인트 준비

필수 원본은 **Teacher FT adapter 약 40MB**와 **Student FT 약 3.09GB**입니다. Teacher를 새로 FT하거나 Student Base로 대체하면 같은 비교가 아닙니다.

```text
원본 서버의 보관 위치(관리자에게 확인):
<SOURCE_CHECKPOINT_DIR>/teacher_ft_best_adapter/
<SOURCE_CHECKPOINT_DIR>/student_ft_best.pt

이 저장소 기본 설정의 도착 위치(프로젝트 기준 상대 경로):
results/imported/azure-qwen7b-v1/teacher_ft_best_adapter/
results/imported/azure-qwen7b-v1/student_ft_best.pt
```

원본 서버의 네트워크 정책에 맞는 승인된 전송 경로를 사용합니다. **private IP만 있는 서버는 외부 PC에서 바로 `scp`로 접근할 수 없습니다.** 승인된 VPN, Azure 사용 시 승인된 Bastion 터널/파일 전송 경로, 또는 조직이 승인한 중계 저장소를 준비하세요. Windows/macOS 모두 같은 원칙입니다.

공용 SSH 방화벽을 열거나 임의로 public IP를 붙이지 않습니다. 토큰·스토리지 키·개인키를 소스에 저장하지 않습니다. adapter는 폴더 전체를 옮기고, 송신/수신 SHA-256과 파일 크기를 대조해 전송 무결성을 확인합니다.

전송 담당자가 완료를 확인한 뒤 `[VESSL 원격]`에서 점검합니다.

```bash
ls -lh results/imported/azure-qwen7b-v1/teacher_ft_best_adapter/
ls -lh results/imported/azure-qwen7b-v1/student_ft_best.pt
bash scripts/run_vessl.sh configs/vessl_qwen7b_korean_3090x4.yaml check
```

adapter 설정 JSON과 가중치(safetensors 또는 bin), Student FT가 모두 있어야 `READY`가 가능합니다. **원본 adapter와 구버전 PEFT 0.13.2의 실제 로딩 호환성은 아직 미확인**입니다. 최초 본 실행의 로딩 단계에서 오류가 나면 중단하고 호환 환경/명시적 변환을 검토합니다. 알 수 없는 adapter 필드를 조용히 삭제해 통과시키지 마세요.

## 5. [VESSL 원격] 본 KD 실행 — 준비 완료 후에만

브랜치/환경 확인, `smoke` 통과, 원본 전송·무결성 확인, `check`의 `READY`를 모두 충족한 뒤 진행합니다. 이때부터 모델/데이터 다운로드와 GPU 계산이 발생합니다.

| 항목 | 유지할 설정 |
|---|---|
| 모델 | Qwen2.5-7B Teacher → Qwen2.5-1.5B Student |
| GPU 배치 | rank 0: Student GPU 0 / Teacher GPU 2; rank 1: Student GPU 1 / Teacher GPU 3 |
| 배치 | rank당 1 × Student 2 ranks = **global batch 2**, Azure H100 기준과 동일 |
| KD | `temperature: 2`, `alpha: 0.5`, `reverse_kl`, `tokenmean` |
| 학습 | LR `1e-5`, Adafactor, sequence 256, 8 epochs, early stopping 꺼짐 |
| 데이터 | 한국어 Wikipedia 고정 revision, 1,000문서, train/val/test 90/5/5% |
| 메모리 | BF16, gradient checkpointing 켜짐; Teacher 가중치는 고정 |

**4개의 GPU가 있다고 Student를 4 ranks로 실행하지 않습니다.** `--nproc-per-node=4` 또는 런처 바깥의 `torchrun`을 사용하지 마세요. DDP는 복제 학습이며 24GB × 4를 하나의 96GB 메모리로 합치지 않습니다.

먼저 원격 설정 파일을 열어 `run_id`가 새 실행용인지 확인합니다. 기본값은 `qwen7b-korean-3090x4-v3-parity`입니다. 재학습이면 예를 들어 `qwen7b-korean-3090x4-v3-parity-r02`처럼 **사용하지 않은 ID**로 바꾸고 저장합니다. 같은 실행의 평가가 끝날 때까지 ID를 유지합니다.

```bash
bash scripts/run_vessl.sh configs/vessl_qwen7b_korean_3090x4.yaml distill
```

런처가 Student 2 ranks를 시작합니다. 각 rank의 Teacher GPU, 원본 adapter/Student FT 로드 성공, 첫 배치와 epoch 로그를 확인합니다. 다른 **원격 터미널**의 `nvidia-smi`로 GPU 메모리를 확인할 수 있습니다. 큰 모델이 OOM이면 성공한 smoke만 근거로 재시도하지 마세요.

**중단/재실행 주의:** 저장되는 것은 best Student 가중치이며 optimizer·진행 step·RNG 재개 상태가 아닙니다. 진정한 checkpoint resume은 지원하지 않습니다. 기존 best가 있으면 런처도 재학습을 막습니다. 삭제로 우회하지 말고 새 `run_id`로 원본 Student FT부터 새 학습을 시작하세요. best가 없는 중단 실행도 별도 ID로 구분합니다.

장시간 실행 전 VESSL 만료 시각과 저장소 보존 정책을 확인합니다. 단순 SSH 창 닫기는 작업 지속을 보장하지 않습니다. 연결을 끊어야 한다면 담당자와 `tmux` 같은 지속 세션을 준비하되 그 안에서도 같은 런처를 사용합니다.

## 6. [VESSL 원격] 전체 test 평가 → 비교

학습이 정상 완료된 뒤 **같은 설정·같은 `run_id`**로 아래 명령을 순서대로 실행합니다.

```bash
bash scripts/run_vessl.sh configs/vessl_qwen7b_korean_3090x4.yaml evaluate
bash scripts/run_vessl.sh configs/vessl_qwen7b_korean_3090x4.yaml compare
```

`evaluate`는 분산 학습과 달리 **단일 프로세스, GPU 0, batch 2**로 전체 test split의 PPL을 평가합니다. Teacher FT / Student KD / 원본 Student FT / Student Base를 순차 로드합니다. 속도 측정은 최대 50배치의 forward 처리량이며 생성 속도나 KD 학습 처리량이 아닙니다.

런처는 학습 시작 시 원본 Student FT를 현재 실행의 비교용 위치에 심볼릭 링크로 연결합니다. 원본을 이동/삭제하면 평가가 깨집니다. 기본 `run_id`의 예상 결과 위치는 다음과 같습니다(이름을 바꿨으면 해당 부분도 달라짐).

```text
results/vessl/checkpoints/qwen7b-korean-3090x4-v3-parity/student_kd_best.pt
results/vessl/logs/qwen7b-korean-3090x4-v3-parity/distill_history.json
results/vessl/logs/qwen7b-korean-3090x4-v3-parity/evaluation_results.json
results/vessl/logs/qwen7b-korean-3090x4-v3-parity/summary.json
results/vessl/figures/qwen7b-korean-3090x4-v3-parity/
```

이번에는 FT를 새로 학습하지 않으므로 이전 FT 학습 이력이 없어 loss 곡선 일부를 건너뛸 수 있습니다. 이것만으로 평가 실패는 아닙니다. 실제 4개 모델의 평가 결과가 있는지 확인합니다.

## 7. 재접속과 자주 발생하는 문제

워크스페이스 재시작 후에는 로컬 컴퓨터에서 접속 정보를 다시 생성합니다. 명령은 [Windows 가이드](vessl-remote-explorer-windows.md) 또는 [macOS 가이드](vessl-remote-explorer-macos.md)의 재접속 단계를 따릅니다.

Remote Explorer의 SSH 목록 새로고침 → **새 Host로 새 창 연결** → 작업 폴더 열기 → 1번의 경로/브랜치 확인을 반복합니다. 재접속은 학습 재개 명령이 아닙니다. 이전 프로세스가 살아 있는지 확인하기 전 중복 실행하지 마세요.

| 증상 | 확인/조치 |
|---|---|
| Host가 안 보임/연결 timeout | workspace가 running인지, 올바른 조직인지 확인하고 SSH 설정 재생성·목록 새로고침 |
| `Permission denied (publickey)` | 현재 계정의 공개키 등록과 생성된 SSH 설정의 키 경로 확인; 개인키 전송 금지 |
| passphrase 요청 | 로컬 보안 입력창/터미널에 직접 입력; 채팅·소스에 기록 금지 |
| 로컬 PC의 Python으로 실행됨 | SSH 표시와 원격 interpreter를 확인; 원격 가상환경 선택 |
| GPU가 1~2개만 보임 | `CUDA_VISIBLE_DEVICES`와 VESSL 할당 확인; CUDA 번호는 노출 후의 **논리 번호**입니다. 할당 밖 GPU를 임의 노출하지 마세요 |
| `WORLD_SIZE`/Teacher 장치 오류 | 4-rank 실행/중첩 torchrun 금지. 깨끗한 원격 터미널에서 이 런처만 실행 |
| `NOT READY`/`BLOCKED` | 종료 코드만 보지 말고 누락 원본·장치 문제를 해결. 경로를 빈 값으로 우회 금지 |
| adapter 알 수 없는 필드 오류 | 원본 보존, PEFT 호환성 검토. 필드 삭제·임의 패키지 업그레이드 금지 |
| `pip check`에서 `pygobject requires pycairo` | 일부 기본 이미지의 GUI 패키지 누락입니다. 이미지 관리자에게 Cairo 개발 라이브러리와 가상환경의 pycairo 설치를 요청하고 재확인합니다. 학습 코드 오류와 구분하세요. |
| CUDA OOM/디스크 부족 | 다른 작업·GPU별 메모리·현재 저장소 할당량 확인. 설정 변경은 별도 실험으로 기록 |
| 비교 결과 없음 | 학습 성공 여부와 같은 `run_id`의 전체 평가 완료 확인 후 `compare` 실행 |

## 8. 결과 해석과 다음 실험

Azure 마지막 완료 v3는 **KD PPL 11.68487 / FT PPL 11.81362**입니다. v4는 준비만 했고 결과 산출물은 없습니다. 과거 H100 수치를 3090 처리량 비교로 사용하지 않습니다.

`seed: 42`는 데이터 추출/분할용이며, **새로 정의한 `training_seed: 42`는 학습 RNG·배치 순서용**입니다. 원래 Azure 학습 RNG는 명시적으로 seed하지 않았습니다. 새 코드에서 Student 1/2 ranks의 global batch 순서를 맞춰도 알 수 없는 과거 순서를 복원하지는 않습니다. 패키지·하드웨어·수치 연산도 달라 **비트 동일 재현, 동일 PPL, 속도 향상은 보장하지 않습니다.**

1. **continued CE 대조군부터:** 동일 Student FT에서 시작해 KD와 update 수·데이터·optimizer 조건을 맞추고 `alpha: 1`과 비교합니다. 추가 학습 효과와 증류 효과를 분리합니다.
2. 다음은 forward/reverse KL만 바꾸는 통제 비교입니다. 이후 LR `5e-6 / 1e-5`, `alpha: 0.5 / 0.7 / 0.9`, T `1 / 2 / 4`를 **한 번에 한 항목씩** 바꿉니다. `alpha`는 CE 비중입니다.
3. 유망 조건은 **training seed 3개**(예: 42/43/44)로 반복하되 데이터 split seed는 42로 고정합니다. validation으로 선택하고 test를 튜닝에 사용하지 않습니다.
4. 10k 데이터 확장은 그다음입니다. 기존 test를 별도로 고정하고 학습 데이터와 겹치지 않게 설계합니다. 단순히 문서 수만 늘리면 같은 split seed여도 test가 바뀔 수 있습니다.

구현 참고: [../src/config.py](../src/config.py), [../src/distributed.py](../src/distributed.py), [../scripts/vessl_3090x4_smoke.py](../scripts/vessl_3090x4_smoke.py).

구현 및 실제 소형 모델 검증 범위: [3090×4 검증 기록](vessl-3090x4-validation.md).