# H100 KD 튜닝 결과 및 개발 인수인계

브랜치: `feature/azure-h100-kd-tuning` (기준 `ae7fa1c`). VESSL 브랜치는 변경하지 않는다.
원본 Azure checkout과 체크포인트는 보존하고 별도 checkout/결과 경로를 사용한다.

## 완료 결과 (2026-09-20)

8개 실험(7 KD + 1 CE-only)과 최종 평가가 16:31 KST에 성공했다. 16:38 KST에 원본 산출물 51개 회수와 로컬 보고서/그림 생성까지 완료했고 감시기는 종료됐다. **2026-09-27 인수인계 시 VM의 현재 전원/보존 상태는 재조회하지 않았다.** 과거 성공 기록을 현재 실행 중으로 해석하지 않는다.

| 모델 | 같은 평가 코드로 재측정한 test PPL |
|---|---:|
| 원본 Student FT | 11.814911 |
| 기존 KD v3 | 11.683636 |
| **새 KD (alpha7)** | **11.353113** |
| CE-only best | 11.814911 (학습 전 step 0 선택) |
| KD와 같은 765 step의 CE-only | 12.068619 |

선택 설정: **LR 1e-5 / alpha 0.7 / T=2 / Reverse KL / tokenmean**, 1 epoch의 step 765. 기존 KD 대비 약 2.83%, 초기 FT 대비 약 3.91% PPL 감소다. 후보는 validation으로만 선택하고 selection을 고정한 뒤 test를 평가했다. 단일 학습 seed의 작은 Wikipedia holdout 결과이므로 통계적 유의성이나 실버케어 일반 성능은 입증하지 않았다.

- [최종 보고서](../results/logs/h100-tuning-20260920/report.md)
- [최종 평가 원본](../results/logs/h100-tuning-20260920/test_results.json), [고정된 선택 결과](../results/logs/h100-tuning-20260920/selection.json)
- [데이터/가중치 해시·패키지 manifest](../results/logs/h100-tuning-20260920/manifest.json), [전체 파라미터 비교 CSV](../results/logs/h100-tuning-20260920/parameter_search.csv)
- [PPL 그래프](../results/figures/h100-tuning-20260920/perplexity_comparison.png), [검증 곡선](../results/figures/h100-tuning-20260920/val_loss.png)

가설과 달리 낮춘 LR은 기존 LR보다 우수하지 않았다. alpha=0.3은 초기 FT보다 개선되지 않았고, alpha=0.7/T=2가 탐색 범위 내 최우수였다. 결과를 본 뒤 같은 test에 맞춰 추가 파라미터를 선택하지 않는다.

## 다른 머신에서 시작하기

```bash
git clone --branch feature/azure-h100-kd-tuning https://github.com/YOONPYOGitHub/knowledge-distillation.git
cd knowledge-distillation
git status --short --branch
```

이미 clone한 **깨끗한 작업 트리**라면 `git fetch origin` 후 `git switch --track origin/feature/azure-h100-kd-tuning`을 사용한다. 로컬 브랜치가 이미 있으면 `git switch feature/azure-h100-kd-tuning` 후 `git pull --ff-only`한다. 다른 미커밋 작업을 reset/clean으로 지우지 않는다. 이 브랜치는 VESSL 브랜치와 병합하지 않았으므로 두 코드가 모두 있다고 가정하지 않는다.

로그와 PNG 열기에는 Python·GPU·Azure 로그인이 필요 없다. 개발/테스트는 Linux 또는 macOS, Windows에서는 **WSL2**를 사용한다. 자동 회수기는 `fcntl`과 POSIX 신호를 사용하므로 Windows 네이티브 Python에서는 지원하지 않는다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
```

CPU 테스트에는 Azure 체크포인트나 GPU가 필요 없다. 확인된 회귀 테스트는 35개다. macOS/Windows 가상환경을 Linux 서버에 복사하지 않는다. 위 requirements는 범위 지정이므로 **과거 학습 환경을 정확히 재현하는 lockfile은 아니다**. H100 학습 패키지 버전은 manifest의 torch=2.11.0+cu128, transformers=4.55.4, datasets=3.6.0, peft=0.20.0, accelerate=1.14.0, numpy=2.5.2이며, CUDA 드라이버/OS에 맞는 환경을 별도로 확인해야 한다. 우선 기존 Azure `/var/lib/kd-venv`를 유지한다.

그림/보고서를 오프라인으로 재생성하려면 저장소 루트에서 실행한다. 모델을 로드하거나 학습하지 않는다.

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.tuning_report import render_tuning_report
summary = render_tuning_report(Path("results"))
print(summary["status"], summary["completed_trials"])
PY
```

원본 JSON의 `/var/lib/...` 경로는 **실험 당시 Azure 경로**다. 로컬로 옮긴 경로로 오해하지 않는다. 생성된 summary/sync 기록의 로컬 절대 경로는 저장 당시의 값일 수 있다. 보고서의 Markdown 링크는 상대 경로이며 위 재생성 명령으로 현재 머신의 파생 경로를 갱신할 수 있다. `notes.md`는 보존된다.

## 코드와 원격 산출물

| 파일 | 역할 |
|---|---|
| [src/h100_tuning.py](../src/h100_tuning.py) | 단일 trial, 초기 평가·분기 epoch 검증·원본 FT 재시작·CE 대조군 |
| [scripts/tune_h100.py](../scripts/tune_h100.py) | 데이터 고정, 8개 순차 탐색, validation 선택 후 최종 test |
| [scripts/azure_h100_tuning.py](../scripts/azure_h100_tuning.py) | 격리 배포, Managed Run Command 제출/상태 조회 |
| [scripts/sync_h100_tuning_results.py](../scripts/sync_h100_tuning_results.py) | 작은 결과 파일 안전 회수·완료 감시 (POSIX) |
| [src/tuning_report.py](../src/tuning_report.py) | 로컬 JSON/CSV/Markdown/PNG 생성 |
| [기본 탐색 설정](../configs/h100_qwen7b_korean_tuning.yaml) | LR 탐색 시작점 alpha=0.5이며 **최우수 설정 파일은 아님** |

**Git에 없는 필수 파일:** Teacher adapter 약 40MB, Student FT 약 3.09GB, 최우수 Student KD 약 3.09GB, Hugging Face 모델 캐시, 토큰화 dataset 디렉터리. 다른 머신에서 clone만 하면 학습을 바로 재개할 수 있다는 뜻이 아니다.

```text
Azure 코드: /var/lib/knowledge-distillation-h100-tuning-20260920
Azure Python: /var/lib/kd-venv/bin/python
결과 루트: /var/lib/kd-results/h100-tuning-20260920
고정 데이터: /var/lib/kd-results/h100-tuning-20260920/data
최우수 KD: /var/lib/kd-results/h100-tuning-20260920/trials/alpha7/best.pt
Teacher: /var/lib/kd-results/qwen7b-4way/checkpoints/qwen7b-korean-4way-v1/teacher_ft_best_adapter
초기 Student FT: /var/lib/kd-results/qwen7b-4way/checkpoints/qwen7b-korean-4way-v1/student_ft_best.pt
기존 KD v3: /var/lib/kd-results/qwen7b-reverse-v3/checkpoints/qwen7b-korean-reverse-v3/student_kd_best.pt
```

VM: `vm-test-100`, resource group `rg-ai`, region `koreacentral`, Spot/Deallocate. 완료 작업 식별자는 [managed_job.json](../results/logs/h100-tuning-20260920/managed_job.json)에 있다. 이 파일은 인증 토큰이 아닌 리소스 식별자다. 새 머신에서 `az login`으로 본인 권한을 인증한 뒤, 해당 구독을 선택하고 다음처럼 **상태만** 조회한다.

```bash
az login
# <SUBSCRIPTION_ID>를 기존 VM이 속한 실제 구독으로 교체
az account set --subscription "<SUBSCRIPTION_ID>"
.venv/bin/python scripts/azure_h100_tuning.py status \
	--subscription "<SUBSCRIPTION_ID>" --resource-group rg-ai --vm-name vm-test-100
```

완료된 실험에 `deploy`/`run --phase all`을 다시 실행하지 않는다. 기존 경로 보호 및 selection 고정 때문에 차단되며, 새 환경 검증 없이 중복 GPU 작업을 만들지 않는다. Spot 중단 시 자동 VM start는 하지 않는다. Azure 원본 디스크 보존·체크포인트 백업을 우선 확인하고, private IP만 있는 서버라면 승인된 VPN/Bastion/중계 저장소를 이용한다. 개인키·토큰은 Git에 넣지 않는다.

## 다음 개발 작업과 제한

1. **최우수 설정을 고정한 학습 seed 반복**(예: 42/43/44): split seed는 42로 고정한다. 현재 orchestrator의 trial seed는 42로 고정되어 있으므로 training seed를 CLI/config로 분리하는 구현이 먼저 필요하다.
2. **새 suite 경로 일반화:** 원격 Python 실행기는 `--config`를 받지만 Azure helper의 `REMOTE/RESULTS`, sync의 `SUITE`, 일부 참조 checkpoint 경로는 20260920에 고정되어 있다. YAML의 output_dir만 바꾸면 helper/회수 경로가 자동 변경되지 않는다. 여러 suite 지원을 먼저 추가하고 원본 결과는 보존한다.
3. **Spot 재개:** 현재는 완료된 trial만 재사용한다. 중단된 trial의 optimizer/RNG/step을 복구하는 진정한 resume은 없다. 실패 디렉터리를 보존하고 새 경로에서 초기 FT부터 재시도하는 절차가 필요하다.
4. **평가 확대:** test를 튜닝에 재사용하지 않고, 독립 한국어·도메인 평가와 데이터 확장 시 문서 중복 없는 고정 holdout을 설계한다.
5. **새 모델 비교:** Qwen3/3.5는 아직 이 브랜치에서 평가하지 않았다. 같은 세대 Teacher/Student tokenizer 호환성·라이브러리·메모리 검증부터 별도 실험으로 진행한다.

이번 인수인계에서는 새 학습·VM 재기동·VESSL 배포를 수행하지 않는다. 아래는 완료된 실험의 원래 설계와 실행 기록이다.

## 목적과 비교 조건

- Teacher: Qwen2.5-7B + 기존 한국어 LoRA FT adapter, 학습 중 고정.
- Student: Qwen2.5-1.5B + 기존 CE-only FT best 가중치에서 **후보마다 재시작**.
- 한국어 Wikipedia 1,000문서, 데이터 revision·분할 seed 42·sequence 256 고정.
- 단일 H100, batch 2, BF16, Adafactor, weight decay 0.01, gradient clip 1 유지.
- 데이터는 한 번 토큰화하여 고정한다. split별 블록 수와 SHA-256을 기록한다.
- 학습 RNG seed 42, epoch별 독립 셔플 RNG를 사용한다. 과거 v3는 학습 RNG를 저장하지 않았으므로 완전히 같은 궤적을 복원하는 실험은 아니다.

## bounded 탐색

1. 실제 7B/1.5B 가중치 + 원본 adapter/FT 체크포인트로 2-update smoke.
2. **LR**: 1e-5, 5e-6, 2e-6 (alpha=0.5, T=2).
3. validation 최우수 LR에서 **alpha** 0.3, 0.7을 추가 비교.
4. validation 최우수 LR·alpha에서 **T** 1, 4를 추가 비교.
5. 선택한 LR에서 **CE-only 추가 학습**(alpha=1)을 동일 초기값·순서·2-epoch 예산으로 수행. 선택 KD의 best step과 동일 step의 CE 가중치도 별도 보존한다.
6. 후보 선택을 파일로 고정한 뒤에만 전체 test를 평가한다. 원래 FT, 기존 v3 KD, 새 KD, CE-only best 및 같은 step CE를 같은 FP32 CE 평가 코드로 재측정한다.

총 7개 KD 후보 + CE 대조군 1개. 각 최대 2 epoch이며 무한 반복하지 않는다.
모든 후보는 validation을 약 1/4 epoch마다 확인하고, **학습 전 step 0 FT도 최종 선택 후보**로 둔다.
최소 validation CE를 선택한다. 초기값보다 나빠진 경우 원본 FT를 반환하며 KD 개선이라고 주장하지 않는다.
초기값을 포함하여 같은 validation 탐색 예산을 CE 대조군에도 적용한다.

## 제어·안전

- 원본 파일은 읽기만 한다. trial 디렉터리가 이미 존재하면 덮어쓰지 않는다.
- 완료된 trial은 같은 plan일 때만 재사용한다. 중단된 trial은 자동으로 재개하지 않는다(optimizer/RNG resume 아님).
- trial 하나 최대 25분, phase 최대 약 83분으로 제한한다. 오류·NaN·디스크 부족 시 탐색을 멈춘다.
- Azure 장시간 실행은 **Managed Run Command**로 하나만 제출한다. 진행 확인은 `status`의 읽기 전용 API로 수행하며 일반 `run-command invoke`를 중복 실행하지 않는다. VM 리전은 조회 결과를 사용한다.
- VM은 Spot/Deallocate일 수 있다. 회수되면 GPU 학습과 프로세스는 중단된다. 실행 전에 전원 상태를 확인하며 자동 VM 재기동은 하지 않는다. 영구 디스크의 완료 result는 보존되지만 미완료 trial은 optimizer/RNG 이어 학습이 불가능하므로 기록을 보존하고 해당 trial만 초기 FT부터 재시도해야 한다.
- 코드 단위 테스트 후 Azure 런타임에서도 테스트한다. 원본 모델 없는 CPU 테스트와 실제 모델 smoke를 구분한다.
- test 점수로 다시 후보를 고르지 않는다. 검증 결과가 좋지 않아도 test 최적화로 전환하지 않는다.
- `warmup_steps`, scheduler, early stopping은 이번 고정-budget 학습기에서 사용하지 않는다. 잘 작동하는 LR을 찾은 뒤 별도 통제 실험으로 다룬다.
- 단일 seed의 개선은 예비 결과다. 유망 조건은 고정 split에서 학습 seed 3개와 도메인/한국어 평가로 추가 확인한다.

## 실행

Azure 원격 전용 코드 경로에서 기존 CUDA 가상환경을 사용한다.

```bash
export HF_HOME=/var/lib/huggingface
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
/var/lib/kd-venv/bin/python scripts/tune_h100.py --phase smoke
/var/lib/kd-venv/bin/python scripts/tune_h100.py --phase lr
/var/lib/kd-venv/bin/python scripts/tune_h100.py --phase alpha
/var/lib/kd-venv/bin/python scripts/tune_h100.py --phase temperature
/var/lib/kd-venv/bin/python scripts/tune_h100.py --phase final
```

기본 결과 루트는 `/var/lib/kd-results/h100-tuning-20260920`이다. 새 suite는 YAML의 output_dir와 함께 Azure helper/회수기의 고정 경로도 수정해야 한다(위 제한 참조).
manifest, plans, trial별 config/result/validation_history, 최종 selection/test_results를 보관한다.
모든 validation CE는 유효 토큰 수로 가중한 FP32 CE이다. 예전 보고의 PPL과 비교할 때는 기존 v3를 동일 평가 코드로 재측정한 값을 함께 사용한다.

위 절차로 완료된 결과는 문서 상단과 실제 result/selection/test_results 파일에 보존되어 있다. 새 trial을 시작하기 전에 반드시 별도 suite와 경로를 준비한다.

## 로컬 결과·이미지 자동 저장

`scripts/sync_h100_tuning_results.py`는 별도 로컬 감시기로 동작한다. 학습을 제출하거나 VM을 재시작하지 않는다.

- 기본 60초마다 Managed Run Command 상태를 읽는다.
- 실행 중에는 5분 간격으로 중간 JSON·로그를 회수하고 **잠정 validation 그래프**를 갱신한다.
- 성공 완료 시 최종 원본 산출물을 해시 검증 후 저장하고 PPL 비교 그림과 요약을 생성한다.
- 로컬 출력 위치는 `--local-results`로 지정한다. 기존 메인 작업 공간의 `results`를 지정하면 과거 실험과 같은 구조로 저장된다.
- 저장 구조: `logs/h100-tuning-20260920/` 및 `figures/h100-tuning-20260920/`.
- 그림: `val_loss.png`, `parameter_search.png`, `training_time.png`; 최종 test가 있으면 `perplexity_comparison.png` 추가.
- 기록하지 않은 학습 loss·추론 속도 그림은 만들지 않는다. 소요 시간에는 로딩·검증·저장이 포함된다.
- 모델 `.pt`, `.safetensors`, dataset cache는 전송하지 않는다. 원본 모델 체크포인트는 Azure에 보존한다.
- Spot 중단 시 기존 로컬 결과를 보존하며 자동 성공 처리하지 않는다. 로컬 Mac 종료/감시기 종료/네트워크 단절 시 자동 회수도 멈추므로 이후 `sync` 또는 `watch`를 다시 실행한다.
- 수동 메모 `notes.md`는 덮어쓰지 않고 자동 보고서는 `report.md`로 갱신한다.