# autoqa 연결 안내 — 팀 작업과 맞물리는 곳

작성 2026-11-08 · 서현범. 모델이 완성되기 전에 정해 두면 나중에 다시 할 일이 없는 것들이다.

## 1. 한눈에 — 누가 무엇을 넘기면 되나

| 누가 | 무엇을 | 언제 | autoqa 가 하는 일 |
|---|---|---|---|
| 모델 담당 (4-way 실험) | 학습이 끝난 run_id (팀 UI 백엔드에 올라간 체크포인트) | run 이 끝날 때마다 | `bash scripts/autoqa_4way.sh <run_id>` 한 줄로 답 받기 → 채점 → 리포트 |
| 문답 KD 담당 (A/B/C 대조군) | 응답 JSONL (아래 §2 형식) 또는 HF 체크포인트 폴더 | 대조군 학습이 끝날 때 | `generate --source hf --model-path <폴더>` 또는 JSONL 그대로 `run` |
| 데이터 담당 | 학습 데이터 파일·폴더 | **학습 전에** | `contamination --train <경로>` 로 골든셋이 섞였는지 검사 (§4) |
| 앱 담당 (가드레일) | 응급 감지 규칙·분류기 | 바꿀 때마다 | `guardrail --heldout …` 으로 처음 보는 응급 발화 감지율 측정 |
| 전문가 검수 (약사·간호사) | 골든셋 문항별 수정 의견 | Proposal 11/23–11/29 | 골든셋 v1 로 올리고 지문이 바뀐 새 회차로 비교 |
| 사용자 평가 (서현범) | 사용자 실험 문항 | 실험 전 | 골든셋 `study_set: true` 12문항에서 10문항 선택 |

## 2. 응답 파일 형식 (계약)

JSONL, 한 줄에 응답 하나.

```json
{"model": "student_kd", "item_id": "SC-E01", "response": "지금 바로 119에 전화하세요. …", "meta": {"run_id": "kanana-8b-2b-v5", "checkpoint": "epoch2"}}
```

| 필드 | 규칙 |
|---|---|
| `model` | 비교 단위 이름. 4-way 는 `teacher_ft` · `student_kd` · `student_ft` · `student_base` (팀 UI 백엔드와 같은 이름). 문답 KD 대조군은 `student_A_goldCE` · `student_B_teacherCE` · `student_C_teacherKL` 처럼 대조군 글자를 앞에 둔다. 양자화본은 뒤에 `_q4` |
| `item_id` | 골든셋 문항 id (`SC-E01` …). **50문항 전부** — 빠지면 prep 이 거부한다(변이 시험처럼 일부러 일부만 넣을 때만 `--allow-partial`) |
| `response` | **모델이 새로 만든 답만.** 프롬프트·시스템 문장·"어르신:" 같은 차례 표식을 넣지 않는다 |
| `meta` | 자유. run_id·체크포인트·생성 설정을 남기면 리포트에서 추적된다 |

넘기기 전에 `python -m silvercare_autoqa check-responses <파일>` 을 돌린다. 이 검사는 빠진 문항·중복·빈 답·시스템 프롬프트 섞임을 잡는다. 팀 UI 백엔드는 prompt 까지 붙여 돌려주므로 실제로 일어나는 실수다.

## 3. 생성 설정 (같은 조건으로 비교하려면)

| 항목 | 값 | 이유 |
|---|---|---|
| 디코딩 | 탐욕(greedy), `do_sample=False` | 같은 입력에 같은 답 → 모델 차이만 남김 |
| 최대 길이 | 160 토큰 | 3문장 답에 충분. `hit_max_tokens` 가 많으면 잘린 답이 많다는 뜻 |
| 반복 억제 | `repetition_penalty=1.1` | 팀 UI 백엔드 기본값과 같음 |
| 시스템 프롬프트 | `silvercare_autoqa/data/app_system_prompt.txt` | 모든 모델이 같은 지시를 받음 |
| 입력 형식 | base 모델 = raw prompt(`어르신: … / 돌봄 AI:`), 대화형(instruct) 모델 = 그 모델의 chat template | `generate --source hf --chat auto` 가 자동으로 고른다. 팀 UI 백엔드는 chat template 을 쓰지 않으므로 대화형 Student 는 `--source hf` 로 받는다 |
| 추론 모드 | 끈다 (`enable_thinking=False`) | 모델 선정 보고서 §8.3 |

## 4. 학습 데이터에서 골든셋 빼기 (학습 전에)

시험 문제를 학습에 넣으면 점수가 부풀어 KD 효과를 잴 수 없다.

```bash
python -m silvercare_autoqa contamination --train data/silvercare_train.jsonl data/teacher_answers/ --out contamination.json
```

- 골든셋 발화·모범 답안·가드레일 시험 발화의 8글자 묶음이 60% 이상 겹치는 학습 문장을 찾는다.
- 하나라도 나오면 종료코드 2다. 그 문장을 빼고 학습한다.
- AI Hub 71762 로 골든셋을 넓힐 때는 같은 원문 질문(fileName)에서 나온 질문을 학습·평가에 나눠 넣지 않는다. 질문 단위가 아니라 원문 단위로 나눈다.

## 5. 지금 바로 할 수 있는 것과 나중에 하는 것

| 지금 (모델 완성 전) | 모델 완성 후 |
|---|---|
| ✅ 연결 시험: 기존 v4 체크포인트로 `autoqa_4way.sh` (점수 의미는 없고 배관 확인) | 도메인·문답 KD Student 4-way 비교 (KD vs FT vs Base vs Teacher) |
| ✅ 지식 주입 전 기준선: 공개 instruct 모델 답 채점 ([VERIFICATION.md](VERIFICATION.md) §8) | 양자화(Q4) 전후 같은 지문으로 비교 |
| ✅ 학습 데이터 오염 검사 규칙 (§4) | 전문가 검수 반영 골든셋 v1 로 최종 수치 |
| ✅ 응답 형식·생성 설정 합의 (§2·§3) | 사람 라벨 40건으로 채점기 일치도(κ) |

## 6. 채점기 실행 조건

- 채점은 `claude -p`(Claude Code, 기본 Sonnet 5.5 · effort high)로 한다. 실행하는 PC에서 `claude` 로그인이 살아 있어야 한다. `python -m silvercare_autoqa preflight --judge` 로 확인한다(1회 호출).
- 서버에 Claude Code 가 없으면 `prep` 까지 서버에서 하고, run 폴더를 Claude Code 가 있는 PC 로 옮겨 `judge → import → verify → finalize` 를 돌린다.
- 채점 모델·골든셋·용어집·룰·프롬프트가 바뀌면 평가기준 지문이 바뀐다. 지문이 다른 회차끼리는 숫자를 비교하지 않는다(`meta.json` 의 `fingerprint`).
