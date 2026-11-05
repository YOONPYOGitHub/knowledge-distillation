# 4-way 협업 인계: 완료된 일반 KD와 진행 중인 한국어 QA

기준: 2026-11-05 KST. 대상: AIminam PR #64 `feature/qwen3-4way-kd` → `develop` 협업자.

**먼저 볼 자료:** [일반 KD 출판 안내](published/generic-kd-v1/README.md) · [공개 결과/메타데이터](../results/published/generic-kd-v1/README.md) · [QA 상태와 출처](published/taskqa-status-20261105.md).
이 문서는 협업을 위한 실행 인터페이스와 실험 규칙이다. 기존 PR의 결과를 대체하거나, QA 학습 완료·서비스 출시를 선언하지 않는다.

## 1. 무엇을 비교하는가

**4-way**는 같은 계열 안에서 다음 네 행을 비교한다는 뜻이다. 모델 네 계열을 뜻하지 않는다.

| 행 | 의미 | 확인할 효과 |
|---|---|---|
| Teacher selected | train으로 적응한 뒤 dev에서 선택·동결한 큰 모델 | 증류할 만한 Teacher인지 |
| Student pretrained | 추가 학습 전 작은 모델 | 출발점 |
| Student CE | 정답 토큰만으로 학습한 같은 Student | 일반 적응 효과 |
| Student KD | 정답 토큰과 Teacher 확률분포를 함께 학습한 같은 Student | **CE 대비** KD 추가 효과 |

- **CE**: 다음 정답 토큰을 맞히는 교차엔트로피 손실. **KD**: Teacher 정보를 Student에 전달하는 지식증류.
- **KL**: 두 확률분포의 차이를 재는 항. **PPL**: 토큰 예측 난이도 지표로 낮을수록 좋다. 대화 품질 점수는 아니다.
- **cohort**: 문서 ID·정제·분할을 함께 고정한 실험 자료 묶음. **seed**: 난수 초기값. **LoRA**: 전체 가중치 대신 작은 추가 파라미터를 학습하는 방법.
- `pretrained → CE`와 `CE → KD`는 다른 효과다. 큰 적응폭을 전부 KD 성과로 계산하지 않는다.

### 완료된 generic 연구

FineWeb2 한국어 평문을 이어 예측하는 일반 계속사전학습(CPT) 연구다. 최종 test에서 Qwen은 세 Student seed 모두 작은 KD 개선, Gemma·Liquid는 개선/악화가 혼재했다. KD의 보편적 우위, 의료안전, 실기기 유용성을 입증한 결과는 아니다. 수치는 위 출판 결과를 기준으로 인용한다.

| 계열 | Teacher | Student | 설정 파일 |
|---|---|---|---|
| Qwen | `Qwen/Qwen3.5-9B` | `Qwen/Qwen3.5-4B` | [`research_qwen35.yaml`](../configs/research_qwen35.yaml) |
| Gemma | `google/gemma-4-12B-it` | `google/gemma-4-E2B-it` | [`research_gemma4.yaml`](../configs/research_gemma4.yaml) |
| Liquid | `LiquidAI/LFM2.5-24B-A2B` | `LiquidAI/LFM2.5-2.6B` | [`research_lfm25.yaml`](../configs/research_lfm25.yaml) |

모델 revision은 아래 그대로 유지한다. Liquid A2B는 활성 규모 표기이지 전체 가중치가 2B라는 뜻이 아니다.

| 모델 | revision |
|---|---|
| Qwen Teacher | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Qwen Student | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Gemma Teacher | `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7` |
| Gemma Student | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| Liquid Teacher | `3fcfdbfa1a1b6d8166846a6323e69f1e1254440d` |
| Liquid Student | `654f9463ce32b05d0429d76fe1f580b27d4c1ac0` |

### PR #64와 숫자를 바로 합치지 않는 이유

검토한 로컬 캐시 `origin/feature/qwen3-4way-kd`의 Gemma3 v3 설정은 `google/gemma-3-12b-pt → google/gemma-3-1b-pt`, Wikipedia `20231101.ko`, revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, 1,000문서·seq256·BOS 추가 조건이다. 같은 캐시의 Qwen3 v4는 `Qwen/Qwen3-8B-Base → Qwen/Qwen3-1.7B-Base`, 10,000문서·seq256·BOS 미추가 조건이다. 이것은 **현재 원격 PR을 다시 조회한 결과가 아니라 로컬 보존본 확인**이다.

이번 일반 연구는 모델 세대·문서 cohort·seq512·학습/선택 예산이 다르다. 과거 표의 PPL, chunk bootstrap, Teacher 격차 감소율을 이번 표와 합쳐 계열 순위나 데이터 규모의 인과효과를 만들지 않는다. 단위 없는 비율도 조건 차이를 없애 주지 않는다. chunk 불확실성과 문서/군집 불확실성, 학습 seed 반복도 구분한다.

## 2. 데이터: 공개 출처와 동일 실험자료는 다르다

### 2.1 일반 연구: FineWeb2 한국어

- 공식 출처: <https://huggingface.co/datasets/HuggingFaceFW/fineweb-2>
- dataset `HuggingFaceFW/fineweb-2`, config `kor_Hang`, 원본 split `train`.
- 고정 revision: `af9c13333eb981300149d5ca60a8e9d659b276b9` (`src/research_data.py`의 `DATASET_REVISION`).
- 정식 공통 cohort: **train 6,814 / validation(dev) 851 / test 851문서**. 공식 데이터셋 test가 아니라 원본 train에서 만든 내부 분할이다.
- canonical `manifest.json` 파일 SHA-256: `53e4e887ec898c2cce985a8027f0dec33360761ba16f705a3c95d244887f904a`.
- 세 계열은 **원문 ID·정제·분할**을 공유한다. 토큰 ID나 토큰 수를 공유한다는 뜻은 아니다.

현재 준비 코드의 규칙은 다음과 같다. 이는 새 자료를 만들 때의 동작 설명이지, 기본 설정으로 역사적 cohort가 다시 만들어진다는 보장이 아니다.

1. 원본 train을 streaming으로 읽고 `shuffle(seed=42, buffer_size=1000)` 후 `max_documents`만큼 취한다.
2. 앞뒤 공백 제거 후 200자 미만, 한글 음절 비율 20% 미만, 이메일/한국 휴대전화 패턴이 있는 문서를 제외한다.
3. 노출 제외 정책이 주어졌으면 `source_ids` 또는 `text_sha256`에 해당하는 문서를 제외하고, 텍스트 SHA-256 exact dedup을 한다. 완전한 개인정보 제거·근접중복 제거는 아니다.
4. `[split seed, text_sha256]`의 digest로 정렬해 validation과 test에 각각 `max(1, 수용문서수 // 10)`개를 배정하고 나머지를 train으로 둔다. Student 학습 seed를 바꿔 재분할하지 않는다.
5. split별로 `add_special_tokens=False`로 인코딩하고 문서마다 EOS를 붙여 seq512로 packing한다. 서로 다른 split은 섞지 않으며 마지막 불완전 블록은 버린다. 별도 BOS를 앞에 추가하지 않는다.

**정확 재현에는 원래 manifest·분할 배정·pilot 노출 제외 정책이 필요하다.** 과거 파일럿에서 노출된 문서를 제외했던 정책을 빠뜨리거나 upstream shuffle을 새로 돌려 얻은 자료를 같은 cohort라고 부르지 않는다. 코드의 제외 정책은 선택 사항이므로 기본 YAML만 실행하면 자동으로 복원되지 않는다. 제외 JSON의 스키마는 `{"source_ids": [...], "text_sha256": [...]}`이며 정책 digest도 identity에 묶인다.

### 2.2 QA 연구: 별도 KLUE-MRC pilot

공식 출처는 <https://huggingface.co/datasets/klue/klue>, config `mrc`, revision `349481ec73fff722f88e0453ca05c77a447d967c`이다. 별도 workspace의 준비 인계기록에 따르면 공식 **train** 17,554행 전체를 기계 스캔한 뒤 문서/중복군집 단위로 1,000문항을 선택했다. 분할은 train **794문항/603군집**, dev **124문항/90군집**, 새 sealed-test **82문항/68군집**이다. 공식 validation/test를 가져와 이 pilot의 test로 쓴 것이 아니다.

- impossible Type3의 비어 있지 않은 원본 answer span은 그럴듯한 오답일 수 있다. 이를 정답으로 학습하지 않으며 `gold_answers=[]`, 빈 gold response와 plausible answer를 구분한다.
- 전체 원본 train을 기계 스캔해 새 분할을 만들었으므로 “선정한 1,000행만 접근” 또는 “아무도 본 적 없는 test”라고 하지 않는다. sealed 재열람·재해시는 이 인계의 절차가 아니다.
- KLUE 카드의 CC BY-SA 4.0만으로 모든 뉴스 재배포·상용화를 허용한다고 해석하지 않는다. Hankyung/Acrofan의 KLUE-MRC·ML 목적 계약, Hankyung의 연구용 조건과 출처표시를 별도 검토해야 한다. 연구 자료를 서비스 지식으로 자동 배포하지 않는다.

### 2.3 협업자가 지금 받을 수 있는 것

| 구분 | 접근/공유 방식 | 사용 범위 |
|---|---|---|
| 공개 결과·집계·출처 설명·공개 코드 | 이 저장소의 출판 안내에서 확인 | 결과 검토와 구현 검토. 원문이나 checkpoint 보유를 의미하지 않음 |
| 원래 generic manifest·split ID/hash 배정·노출 deny 정책·frozen plan | 공개된 [canonical metadata manifest](../results/published/generic-kd-v1/cohort-manifest.json)와 [선택 metadata](../results/published/generic-kd-v1/selected-metadata.json)를 먼저 확인. 전체 문서별 split 배정·역사적 frozen plan/capsule 등 추가 자료는 원 소유자에게 승인된 범위로 요청 | 공개 metadata는 원문 분할 파일이나 전체 역사적 실행 패키지를 대신하지 않음 |
| 원래 정제 corpus·heldout·가중치/adapter | Git 배포 대상 아님. 별도 승인·접근 통제 필요 | 정확한 재학습/재채점은 이것들과 원 환경 계약 없이는 불가능 |
| upstream에서 새로 취득한 자료 | 공식 출처와 고정 revision을 사용하고 새 manifest 발급 | **새 cohort** 실험. 기존 cohort의 정확 재현으로 보고하지 않음 |
| QA pilot·원문/정답·Teacher 응답·운영 capsule | 별도 비공개 workspace, 권리/접근 승인 필요 | 현 저장소 fresh clone만으로 운영 QA 전체를 재현할 수 없음 |

원문·test·모델 파일을 PR에 올리지 않는다. 메타데이터도 URL/ID에 개인정보가 없는지 확인하고 공개 승인된 범위만 올린다.

## 3. 함께 고정할 최소 실험 계약

아래는 **완료된 일반 연구의 계약**이다. QA로 옮길 때도 통제 원칙은 유지하되 손실 mask·scorer·예산은 새 계약으로 고정한다.

| 항목 | 공통 규칙 / 기록할 값 |
|---|---|
| 데이터 | 같은 canonical 문서 ID·정제·분할·제외 정책. 학습 seed 변경으로 split 재생성 금지 |
| 모델/토큰화 | 위 revision 고정. 계열 내 Teacher/Student 전체 token→ID map, backend, special token, 한국어 인코딩 일치 검사. vocab 크기만 같다고 통과시키지 않음 |
| 초기화 | 각 Student seed 안에서 CE/KD 모두 같은 pretrained+LoRA 초기값. CE 학습 결과에서 KD를 시작하지 않음 |
| Teacher | train 적응 후 validation best 선택·동결. 개선이 없으면 step0 유지 가능. 계열당 Teacher 하나 고정 |
| 공통 학습 | LoRA rank8/alpha16/dropout0.05, batch1, seq512, AdamW(weight_decay0.01). LoRA alpha와 아래 손실 alpha는 다른 값 |
| 손실/정밀도 | next-token causal shift, FP32 CE 손실합/유효 토큰수. BF16 backbone과 native FP32 LoRA 저장 구분. KD는 검증한 tokenizer ID support 사용, 무관한 vocab 임의 절단 금지 |
| 탐색 | 계열당 두 round, round당 CE4/KD4 후보. CE도 독립 LR 탐색. 후보당 최대3,000 update, validation200/checkpoint100, patience3/min_delta0.0001 |
| 선택 | dev만으로 후보와 best checkpoint 선택. 최대 update 도달은 수렴 증명이 아님. Teacher 추론이 추가되므로 equal-update ≠ equal-cost |
| 반복 | 설정 고정 후 Student seed42/43/44. seed42는 선택 때의 실행 재사용, seed43/44가 추가 반복. Teacher 세 번 반복 아님 |
| 수치정책 | frozen plan의 deterministic/TF32/CUBLAS 정책 보존. 실제 선택 Student 기록은 deterministic=true, cuDNN deterministic=true/benchmark=false, TF32=false, CUBLAS `:4096:8` |
| 최종 채점 | 선택·cohort·checkpoint·실행 source를 고정한 뒤 test 채점. 이번 공통 test 결과는 이미 공개되어 새 blind tuning에 재사용 불가 |

완료 선택쌍의 CE/KD LR은 Qwen `8e-5`, Gemma `1.5e-4`, Liquid `1.25e-4`였다. KD는 모두 forward KL·온도1.0, **CE 가중치** alpha는 각각0.7/0.7/0.9다. Teacher LR은 `2e-5`였다. 이 값들은 탐색 결과이며 저장소 smoke YAML 기본값과 다르다.

평가는 손실합과 유효 토큰수를 함께 보존한다. 상대 PPL 변화는 `100 × (PPL_KD / PPL_CE − 1)`로 음수가 개선이다. 서로 다른 tokenizer는 경계·tail·예측 대상 byte coverage가 달라 **raw PPL, 근사 BPB, Teacher gap closure만으로 계열 순위를 만들지 않는다.** cross-family 비교에는 동일 한국어 과제·문항·문맥·scorer와 coverage 검증이 필요하다. 문서별 손실이 없다면 aggregate만으로 문서 신뢰구간을 새로 만들지 않는다.

## 4. 공개 코드와 실행법

### 4.1 설치는 4-way 환경과 분리

기존 `pyproject.toml`은 Transformers `<5.0.0`과 구형 tokenizers 범위를 사용하지만 이번 연구 실행 환경의 Transformers는 **5.20.2**였다. 최신 text model class를 쓰는 연구 코드를 기존 4-way venv에 덮어 설치하거나 `uv sync`로 한꺼번에 해결하지 않는다.

저장소 루트에서 별도 Python 3.10 이상으로 실행한다. 아래 requirements는 연구용 별도 설치 명세이며, 과거 VM 전체 환경/운영 capsule의 완전한 복제 보장은 아니다. 파일이 없는 부분 체크아웃에서는 출판 담당자에게 먼저 확인한다.

```bash
python3 --version
python3 -m venv .venv-research
.venv-research/bin/python -m pip install -r requirements/research-2026.txt
.venv-research/bin/python -m pip check

# 데이터·토크나이저·모델을 가져오지 않는 CLI 확인
.venv-research/bin/python -B scripts/run_research.py --help
.venv-research/bin/python -B scripts/tune_research.py --help
.venv-research/bin/python -B scripts/evaluate_final.py --help
```

CUDA/PyTorch 호환성·모델별 사용약관/gating·실제 GPU 메모리는 실행 전에 별도 확인한다. `.venv-research`를 쓴다는 것만으로 Mac에서 A100 BF16 실행이 재현되지는 않는다. `pip install -e .`로 기존 project dependency를 섞지 않는다.

### 4.2 어떤 코드가 무엇을 하는가

| 진입점 | 역할 / 주의 |
|---|---|
| [`scripts/run_research.py`](../scripts/run_research.py) | `prepare → teacher → ce → kd`의 작은 단계별 실행. `evaluate`와 `all`은 test 채점까지 이어지므로 개발 기본명령으로 사용하지 않음 |
| [`scripts/tune_research.py`](../scripts/tune_research.py) | 완료 Teacher와 고정 자료를 가진 `--base-run`에서 별도 validation-only sweep. 단, base raw 파일 SHA 검증은 test 파일도 읽음 |
| [`scripts/evaluate_final.py`](../scripts/evaluate_final.py) | 별도 동결 bundle·SHA·source/선택/cohort 결속으로 최종 채점. bundle 경로만 복사한 새 clone에서 바로 실행 불가 |
| [`src/research_data.py`](../src/research_data.py), [`research_models.py`](../src/research_models.py), [`research_runner.py`](../src/research_runner.py), [`research_tuning.py`](../src/research_tuning.py) | 데이터 identity·공식 text loader·LoRA·수치 손실·체크포인트·탐색의 연구 핵심 구현 |

공개 코드는 이식 가능한 연구 helper와 회귀검사다. 모든 역사적 VM 서비스·인증·storage·절대경로 고정 capsule이나 별도 taskQA 운영 구현이 들어 있다는 뜻은 아니다.

### 4.3 새 smoke 실행: 정식 결과의 재현으로 부르지 않기

다음은 **데이터 취득/모델 접근이 허가되고 CUDA 자원을 준비한 협업자용 예시**다. 이 문서 작성에서는 실행하지 않았다. 기존 run이나 공개 test를 재튜닝하지 않고 새 로컬 경로를 택한다.

```bash
# 저장소 루트에서 실행. 이 명령의 smoke는 200 원본문서/최대20 update 설정이다.
RUN_DIR="$PWD/results/local-qwen35-smoke-new"
.venv-research/bin/python scripts/run_research.py prepare \
  --config configs/research_qwen35.yaml --run-dir "$RUN_DIR"
.venv-research/bin/python scripts/run_research.py teacher \
  --config configs/research_qwen35.yaml --run-dir "$RUN_DIR"
.venv-research/bin/python scripts/run_research.py ce \
  --config configs/research_qwen35.yaml --run-dir "$RUN_DIR"
.venv-research/bin/python scripts/run_research.py kd \
  --config configs/research_qwen35.yaml --run-dir "$RUN_DIR"
```

- `prepare`도 upstream 데이터와 양쪽 tokenizer를 취득한다. 원문 정제 중 내부 test를 만들고 재사용 시 무결성 SHA를 읽지만, train/dev만 packing한다. 내용 비접근 보장이나 공식 blind test 생성 도구로 해석하지 않는다.
- 제외 정책이 필요한 새 cohort에는 **첫 prepare부터 모든 단계에서 동일한** `--exposure-exclusion-file "$EXCLUSION_FILE"`을 붙인다. `EXCLUSION_FILE`은 승인받은 실제 정책 파일이어야 한다.
- 동일 run-dir의 config/identity가 다르면 실패하는 것이 정상이다. 예산·자료·모델을 바꿀 때는 새 run-dir를 쓴다. 중단 재개는 원 계약 그대로 `--resume`을 추가하며, 재개를 새 독립 실행으로 세지 않는다.
- Gemma/Liquid는 config를 바꾸고 별도 run-dir를 쓴다. **각자 prepare해 얻은 문서가 같다고 추정하지 말고** manifest/분할을 확인한다. 공통 비교는 먼저 하나의 승인된 canonical cohort를 고정해야 한다.

### 4.4 고정 Teacher에서 sweep / 최종 평가 인터페이스

아래 환경변수는 자동 발견되는 경로가 아니다. 담당자가 검증한 실제 artifact를 지정해야 하며 비어 있으면 명령이 중단된다. 종료된 출판 cohort의 추가 탐색 지시가 아니라 **새로 승인된 실험의 CLI 예시**다.

```bash
: "${BASE_RUN:?완료 Teacher와 frozen config/prepared/raw를 가진 실제 경로 필요}"
: "${SWEEP_DIR:?새 sweep 출력 경로 필요}"
: "${TRIALS_FILE:?사전에 고정한 균형 CE/KD 후보 JSON 필요}"
.venv-research/bin/python scripts/tune_research.py \
  --base-run "$BASE_RUN" --output-dir "$SWEEP_DIR" \
  --trials-file "$TRIALS_FILE" --seed 42 --max-steps 3000 \
  --validation-interval 200 --checkpoint-interval 100 --patience 3
```

후보 JSON은 각 행에 `id`, `method`, `learning_rate`, `alpha`, `temperature`, `divergence`를 가진 list다. CE/KD 수가 같고 총2–8개여야 하며 CE의 alpha는1, KD는1 미만이다. 기본 후보는 역사적 두 round의 최종 frozen 후보 전체를 대신하지 않는다. seed43/44에서는 **이미 선택한 CE1/KD1 설정만** 고정해 반복하고 다시 후보를 고르지 않는다. `--base-run`에는 Teacher 완료기록과 원래 raw/prepared identity가 필요하므로 model ID만으로 시작할 수 없다.

최종 채점의 실제 옵션은 다음과 같다. 기존 최종 test를 다시 열 권한을 이 예시가 부여하지 않는다.

```bash
: "${BUNDLE:?독립 검토/승인된 frozen 평가 bundle 실제 경로 필요}"
: "${BUNDLE_SHA256:?외부에서 고정한 정확한 bundle SHA256 필요}"
.venv-research/bin/python scripts/evaluate_final.py \
  --bundle "$BUNDLE" --bundle-sha256 "$BUNDLE_SHA256" --device cuda
```

bundle은 절대 source 경로·source SHA·선택 checkpoint·cohort·작업 identity를 결속한다. 새 clone에서 경로가 달라지면 검토된 새 실행 결속이 필요하며 hash 검사를 끄거나 임의의 bundle/hash를 만들어 통과시키지 않는다. 작은 `run_research.py evaluate` 결과의 smoke 표기를 정식 최종 평가로 승격하지 않는다.

## 5. 진행 중 QA: 다음 비교는 A/B/C

QA는 평문 PPL을 더 낮추는 연구가 아니라 **주어진 근거로 답하고, 답할 수 없으면 유보하는 능력**을 평가한다. EM은 정확일치, F1은 정답 중첩 지표다. 이 pilot의 동결 scorer는 공식 KLUE EM/ROUGE-W 점수와 같다고 부르지 않는다.

완료된 사전학습 QA dev124(답변가능93/불가능31, 90군집)의 전체 EM은 Teacher **79/124**, Student **82/124**로 Teacher 우위 gate가 false였다. abstain-first 출력순서는 불가능 문항 거짓답변을 줄였지만 과잉거절과 답변가능 성능 하락이 컸다. 따라서 큰 Teacher나 형식에 맞는 JSON을 자동으로 좋은 학습 표적이라 판단하지 않는다.

후속 Teacher QA 적응은 별도 진행 경로다. 인계된 **2026-11-05 00:34 KST 저장 관측**은 step600/best500 후 중단 상태였고 Student A/B/C는0건이었다. 이 문서 작성은 원격/GPU 상태를 새로 점검하지 않았으므로 “현재 학습 중/정지 중”이나 완료 성능을 단정하지 않는다. 이후 상태는 [날짜가 붙은 QA 상태 보고](published/taskqa-status-20261105.md)를 따른다. Teacher 적응 중간 step은 QA KD 효능 결과가 아니다.

| arm | 학습 표적/손실 | 식별하려는 효과 |
|---|---|---|
| A | 독립 gold 응답의 response-only CE | 과제 정답 학습 대조군 |
| B | 우위/품질 gate를 통과한 검수 Teacher 응답의 CE | 응답 데이터가 바뀐 효과 |
| C | **B와 완전히 동일한 응답**의 CE + KL | **B→C만** KL 추가 효과 |

같은 Student 초기값·문항/군집 분할·response mask·선택 규칙·예산을 유지한다. Teacher 적합성과 답변 검수에 통과하기 전에는 B/C를 자동 진행하지 않는다. 기존 pilot 계획은 처음 각1,000 update 상한·1 seed이며 완료/수렴 보장이 아니다. on-policy, 새 출력순서, 새 데이터 규모를 동시에 바꾸지 않는다. 반복 노출된 dev를 독립 최종증거로 부르지 않으며 sealed test로 설정을 고르지 않는다.

**현재 공개 범위는 결과/설계/핵심 generic 코드다.** taskQA는 별도 workspace에서 진행됐고 이 저장소의 fresh clone만으로 production QA를 실행할 수 있다고 약속하지 않는다. QA 효과·양자화·실기기 RAM/지연/발열·의료안전·실사용 효용은 각각 별도 검증 항목이다.

## 6. 협업자가 보내 줄 최소 결과와 전달 문구

PR에는 모델/revision, cohort manifest/제외 정책 hash, 선택 전 동결한 후보/예산, 각 seed의 초기 adapter hash, Teacher 선택, dev best/latest step, 최종 선택/채점 identity, 손실합/유효 토큰수, CE 대비 KD 변화와 실패/중단 사유를 남긴다. 원문·heldout·가중치는 첨부하지 않는다. 같은 자료인지 확인 전에는 기존 표에 행을 추가하지 않는다.

> PR #64의 4-way 구조와 연결해 볼 수 있도록 일반 KD 3계열 결과와 재현 규칙을 정리했습니다. 이번 결과는 Qwen3.5/Gemma4/Liquid와 FineWeb2 공통 문서 조건이며 기존 Wikipedia/Gemma3·Qwen3 조건과 숫자를 직접 합치지는 않습니다. 먼저 공개 결과와 데이터 manifest/노출 제외 정책의 공유 가능 범위를 함께 확인한 뒤, 같은 Student 초기값·공정한 CE/KD 탐색·dev 선택 규칙을 맞추면 좋겠습니다. QA는 Teacher 적합성 확인과 A/B/C 대조를 진행하는 별도 연구로, 아직 KD 효능 완료 결과가 아닙니다.

---

**작성 검증 범위:** 로컬 코드·설정·기존 일반 통합보고, QA 보고 초안/준비 인계/권리/지속 승인 기록 및 로컬 PR branch 캐시를 읽었다. 세 CLI의 `--help`는 실제 exit0을 확인했다. 이 문서 작성을 위해 corpus·heldout·모델/응답 payload를 열거나 모델을 실행하지 않았고, cloud 조작·의존성 설치·Git staging/commit은 하지 않았다. 위 학습/설치 예시는 코드 인터페이스 기반 안내이지 이번 작성에서 재학습한 결과가 아니다. 역사적 모든 test 비접근을 증명한다는 주장도 하지 않는다.
