# generic-kd-v1과의 차이 공개

소형 모델 3개 조합(Qwen3, Gemma 3, Kanana)을 `generic-kd-v1`과 같은 절차로 평가하면서 달라진 점을 기록합니다. 평가가 진행되면 이 문서도 함께 갱신합니다.

## 같은 조건

- **데이터:** FineWeb2 내부 cohort. train·validation·test 파일의 SHA-256이 `results/published/generic-kd-v1/cohort-manifest.json`과 일치합니다.
- **학습:** seq512, BOS 없음, 문서별 EOS, LoRA rank8/alpha16/dropout0.05, batch1, AdamW weight decay 0.01, Teacher LR 2e-5, 최대 3,000 update, validation 200, checkpoint 100, patience 3, min_delta 1e-4.
- **탐색:** 계열당 2라운드, 라운드당 CE 4개와 KD 4개. 1라운드 후보는 `default_trials()`로 동일합니다.
- **지표:** FP32 CE 합계 / 유효 토큰, CE 대비 KD 상대 PPL 변화. seed 42·43·44.

## 다른 조건

| 항목 | generic-kd-v1 | small-model-v1 | 이유와 영향 |
|---|---|---|---|
| GPU | A100 Spot | RTX 3090 24GB × 4 | 가용 자원. 수치가 비트 단위로 재현되지 않습니다. 환경 차이는 측정하지 않았습니다. |
| Transformers | 5.20.2 (연구 환경 기준) | 4.51.3 | Kanana 설정이 5.x 검증을 통과하지 못합니다. |
| PyTorch | 2.11.0+cu128 (보존 probe 기준) | 2.3.1+cu121 | 4.51.3 환경에 맞춘 값입니다. |
| 모델 변형 | Qwen3.5 일반판, Gemma 4 `-it`, LFM2.5 | Qwen3 일반판, Gemma 3 `-it`, Kanana `instruct-2505` | 서비스 후보로 쓸 수 있는 대화용 변형입니다. |
| Gemma 3 Teacher 배치 | 단일 GPU | GPU 2장 분할 | 가중치 24.37GB로 3090 한 장에 들어가지 않습니다. KD에서는 Student가 첫 번째 GPU를 함께 씁니다. |
| 2라운드 설계 | 1라운드 결과를 본 뒤 계열별로 설계 | [사전 고정 규칙](ROUND2_RULE.md) | 모든 조합에 같은 규칙을 적용합니다. |
| 최종 채점 도구 | `scripts/evaluate_final.py` | `scripts/fineweb/evaluate_test.py` | 기존 도구는 동료 3개 계열만 지원합니다. test 패킹(`load_final_heldout_batches`)과 CE 계산(`final_token_ce_statistics`, `aggregate_final_token_ce`)은 동료 코드를 그대로 씁니다. 모델 dtype 검사는 이 환경의 PEFT에 `lora_variant`가 없어 DoRA 미사용 여부로 같은 조건을 확인합니다. 같은 도구로 validation을 채점해 학습 기록과 소수점 넷째 자리까지 일치함을 확인한 뒤 test에 썼습니다. |

## 코드 변경

동료 브랜치 `origin/feature/azure-h100-kd-tuning`(`dbdaa7d`)의 코드에서 다음만 바꿨습니다.

- `src/research_models.py`: `qwen3`, `llama`, `gemma3` 로더 추가. `dtype=`를 4.51.3이 받는 `torch_dtype=`로 변경. 설정에 `device_map`이 있으면 분할 로딩. 4.52 미만에서는 `key_mapping`을 쓰지 않습니다. 4.51.3이 `language_model.` 접두사를 직접 처리해, 이름을 먼저 바꾸면 모든 텐서를 건너뛰기 때문입니다. 수정 후 Gemma 3 12B의 텍스트 텐서 전체가 파일과 일치함을 확인했습니다.
- `src/research_runner.py`: 분할 로딩된 모델은 `.to(device)`를 건너뜁니다.
- `src/research_tuning.py`: KD 범위를 Student가 예측할 수 있는 토큰 ID로 제한합니다. Gemma 3는 토크나이저가 262,145개, 1B Student 출력이 262,144개이고, 남는 ID 262144는 이미지 자리표시 토큰입니다. train·validation에 그 ID가 없음을 확인한 뒤에만 262,144로 정하고, 있으면 실패합니다. Qwen3와 Kanana는 값이 바뀌지 않습니다.
- `scripts/run_research.py`: `prepare` 성공 후 인터프리터 종료 과정을 건너뜁니다. 스트리밍 스레드가 종료 중 `PyGILState_Release`로 충돌한 문제를 막습니다. 결과 파일은 그 전에 fsync됩니다. 수정 후 새 준비 3회 모두 정상 종료했고 파일 SHA가 게시값과 일치했습니다.

## 폐기한 시도

Qwen3와 Kanana의 Base 변형 Teacher를 먼저 학습했으나, 서비스 후보로 쓰기 어려워 대화용 변형으로 전환했습니다. 두 Teacher의 validation PPL은 Qwen3 7.054→6.765, Kanana 8.195→6.839였고, 이 값은 결과 비교에 쓰지 않습니다. 전환은 후보 탐색과 test 채점 전에 결정했습니다.

## 측정하지 않은 것

- A100과 3090의 환경 차이. 작은 차이는 우열로 해석하지 않습니다.
- 양자화, 실기기 메모리·지연, QA, 안전 평가.

## 중단 기록

- **2026-11-07, Qwen3·Kanana Teacher:** `No space left on device`로 중단되어 마지막 완전한 체크포인트(Qwen3 step 500, Kanana step 700)에서 재개했습니다. `/root`는 다른 워크스페이스와 함께 쓰는 537GB 볼륨의 하위 폴더이고, 이 볼륨이 가득 찼습니다. 이 워크스페이스의 100GB 할당량에는 여유가 있었습니다. 재개는 새 실행으로 세지 않습니다.
- **2026-11-07, Gemma 3 Teacher:** 가중치 다운로드가 끊겨 학습 시작 전에 실패했습니다. 이후 모든 가중치를 SHA-256으로 검증한 뒤 시작합니다.

모델 가중치는 컨테이너 로컬 디스크에 두고, 체크포인트와 결과는 `/root`에 둡니다.
