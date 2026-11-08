# autoqa 연결 안내 — 팀 작업과 맞물리는 곳

## 한눈에

- **모델 담당**은 학습이 끝난 run_id 만 넘기면 된다. `bash scripts/autoqa_4way.sh <run_id>` 가 점검부터 리포트까지 한다.
- **응답 파일**로 넘길 때는 50문항 전부, 모델이 새로 만든 답만 넣는다 (§2). 팀 4-way 이름표도 §2 에 있다.
- **데이터 담당**은 학습 전에 `contamination` 으로 골든셋이 섞였는지 본다 (§4).
- 같은 조건으로 비교하려면 생성 설정을 맞춘다 (§3). 다르면 리포트에 "⚠ 생성 조건 다름"이 붙는다.
- 전문가 채점·κ·어르신 실험은 §7 이다.

작성 2026-11-08 · 서현범. 모델이 완성되기 전에 정해 두면 나중에 다시 할 일이 없는 것들이다.

## 1. 누가 무엇을 넘기나

| 누가 | 무엇을 | 언제 | autoqa 가 하는 일 |
|---|---|---|---|
| 모델 담당 (4-way 실험) | 학습이 끝난 run_id (팀 UI 백엔드에 올라간 체크포인트) | run 이 끝날 때마다 | `bash scripts/autoqa_4way.sh <run_id>` 한 줄로 점검 → 답 받기 → 채점 → 리포트 |
| 문답 KD 담당 (A/B/C 대조군) | 응답 JSONL (아래 §2 형식) 또는 HF 체크포인트 폴더 | 대조군 학습이 끝날 때 | `generate --source hf --model-path <폴더>` 또는 JSONL 그대로 `run` |
| 데이터 담당 | 학습 데이터 파일·폴더 | **학습 전에** | `contamination --train <경로>` 로 골든셋이 섞였는지 검사 (§4) |
| 앱 담당 (가드레일) | 응급 감지 규칙·분류기 | 바꿀 때마다 | `guardrail --heldout …` 으로 처음 보는 응급 발화 감지율 측정 |
| 전문가 검수 (약사·간호사) | 블라인드 채점지 | Proposal 11/23–11/29 | `export-blind` 로 채점지를 만들고, 채운 표를 `import-labels` 로 반입 (§7) |
| 사용자 평가 (서현범) | 4-way 회차 | 실험 전 | `study-materials` 로 어르신 실험 묶음 ([실행안](../user_study/README.md)) |

## 2. 응답 파일 형식 (계약)

JSONL, 한 줄에 응답 하나.

```json
{"model": "student_kd@s42", "item_id": "SC-E01", "response": "지금 바로 119에 전화하세요. …",
 "meta": {"run_id": "kanana-8b-2b-v5", "prompt_format": "raw", "temperature": 0.0, "max_new_tokens": 160, "repetition_penalty": 1.1}}
```

| 필드 | 규칙 |
|---|---|
| `model` | 비교 단위 이름. 아래 이름표를 따른다. seed 를 반복했으면 `조건@seed` (예: `student_kd@s42`) — 리포트가 조건별로 묶는다 |
| `item_id` | 골든셋 문항 id (`SC-E01` …). **50문항 전부** — 빠지면 prep 이 거부한다(변이 시험처럼 일부러 일부만 넣을 때만 `--allow-partial`) |
| `response` | **모델이 새로 만든 답만.** 프롬프트·시스템 문장·"어르신:" 같은 차례 표식을 넣지 않는다 |
| `meta` | 자유. 다만 생성 조건(`prompt_format`·`system_sha`·`temperature`·`max_new_tokens`·`repetition_penalty`)을 남기면 조건이 다른 모델끼리의 비교에 경고가 붙는다. 생성이 실패한 답은 빈 답 대신 `"error": "…"` 를 적는다 |
| `sample` | 쓰지 않는다. 한 문항에 답 여러 개는 아직 집계 규칙이 없어 prep 이 거부한다 |

**팀 4-way 이름표** (팀 문서 `FOUR_WAY_REPRODUCTION_GUIDE.md` 의 행과 같은 뜻)

| 팀 문서의 행 | autoqa `model` | 보는 효과 |
|---|---|---|
| Teacher selected | `teacher_ft` | 증류할 만한 Teacher 인지 |
| Student pretrained | `student_base` | 출발점 |
| Student CE | `student_ft` | 일반 적응 효과 |
| Student KD | `student_kd` | **CE 대비** KD 추가 효과 (`--pairs student_kd:student_ft`) |
| 문답 KD A/B/C | `student_A_goldCE` · `student_B_teacherCE` · `student_C_teacherKL` | B→C 만 KL 추가 효과 |
| 양자화본 | 뒤에 `_q4` (예: `student_kd_q4@s42`) | 양자화 전후 |

seed 반복이 있으면 짝 비교는 조건 이름으로 준다 (`--pairs student_kd:student_ft`). 같은 seed 끼리 짝지어 seed 마다 비교하고 평균±표준편차를 낸다. 조건 판정은 **가장 나쁜 seed** 를 따른다 — 가장 좋은 seed 하나만 보고하면 효과가 부푼다.

넘기기 전에 `python -m silvercare_autoqa check-responses <파일>` 을 돌린다. prep·run 도 같은 점검을 하고, 빠진 문항·빈 답·시스템 프롬프트 섞임·차례 표식·생성 오류가 있으면 채점하지 않는다. 팀 UI 백엔드는 prompt 까지 붙여 돌려주므로 실제로 일어나는 실수다. Windows 에서 저장한 BOM·cp949 파일도 읽는다.

## 3. 생성 설정 (같은 조건으로 비교하려면)

| 항목 | 값 | 이유 |
|---|---|---|
| 디코딩 | 탐욕(greedy), `do_sample=False` | 같은 입력에 같은 답 → 모델 차이만 남김 |
| 최대 길이 | 160 토큰 | 3문장 답에 충분. `hit_max_tokens` 가 많으면 잘린 답이 많다는 뜻 |
| 반복 억제 | `repetition_penalty=1.1` | 팀 UI 백엔드 기본값과 같음. OpenAI 호환 서버에도 같이 보낸다 |
| 시스템 프롬프트 | `silvercare_autoqa/data/app_system_prompt.txt` | 모든 모델이 같은 지시를 받음 (`system_sha` 로 기록) |
| 입력 형식 | base 모델 = raw prompt(`어르신: … / 돌봄 AI:`), 대화형 모델 = 그 모델의 chat template | `generate --source hf --chat auto` 가 고르고, 고른 형식을 출력·기록한다 |
| 양자화본 (llama.cpp·vLLM) | base 모델은 `--source openai --openai-mode raw`, 대화형은 `--openai-mode chat` | 양자화 전후를 **같은 입력 형식**으로 받아야 차이가 양자화 탓이 된다 |
| 추론 모드 | 끈다 (`enable_thinking=False`) | 모델 선정 보고서 §8.3. 생각 과정이 섞이면 형식 Fix |

- base 모델 tokenizer 에도 chat template 이 들어 있는 경우가 있다. `--chat auto` 가 출력한 형식을 꼭 확인한다.
- 팀 UI 백엔드는 chat template 을 쓰지 않는다. 대화형 Student 는 `--source hf` 로 받는다.
- 한 회차 안에서 모델마다 입력 형식·디코딩이 다르면 check-responses·prep 이 경고한다.
- 리포트의 짝 비교·Teacher 비교 줄에는 "⚠ 생성 조건 다름"이 붙는다.

## 4. 학습 데이터에서 골든셋 빼기 (학습 전에)

시험 문제를 학습에 넣으면 점수가 부풀어 KD 효과를 잴 수 없다.

```bash
python -m silvercare_autoqa contamination --train data/silvercare_train.jsonl data/teacher_answers/ --out contamination.json
```

- 골든셋 발화·모범 답안·가드레일 시험 발화의 8글자 묶음이 60% 이상 겹치는 학습 글을 찾는다. 이어진 5줄을 함께 보므로 문장마다 한 줄로 나눠 넣은 것도 잡는다. JSON 줄은 풀어서 보므로 `json.dumps` 기본값(\uXXXX)으로 저장한 한글도 잡는다.
- 확장자와 상관없이 글자 파일은 모두 읽는다. 압축·parquet 처럼 못 읽는 파일은 '문제'로 남기고 실패한다 — 먼저 jsonl·txt 로 풀어 넣는다.
- 하나라도 나오면 종료코드 2다. 그 문장을 빼고 학습한다.
- **결과 파일에는 학습 문장 원문을 넣지 않는다** (위치·문항·겹친 비율만). 학습 데이터에는 반출 금지 원문(AI Hub 71762)이 섞이기 때문이다. 원문을 봐야 하면 `--show-text` — 이 PC 에서만 보고 커밋·공유하지 않는다. `contamination*.json` 은 git 이 무시한다.
- AI Hub 71762 로 골든셋을 넓힐 때는 같은 원문 질문(fileName)에서 나온 질문을 학습·평가에 나눠 넣지 않는다. 질문 단위가 아니라 원문 단위로 나눈다.

## 5. 지금 바로 할 수 있는 것과 나중에 하는 것

| 지금 (모델 완성 전) | 모델 완성 후 |
|---|---|
| ✅ 연결 시험: 기존 v4 체크포인트로 `autoqa_4way.sh` (점수 의미는 없고 배관 확인) | 도메인·문답 KD Student 4-way 비교 (KD vs CE vs pretrained vs Teacher, seed 별) |
| ✅ 지식 주입 전 기준선: 공개 instruct 모델 답 채점 ([VERIFICATION.md](VERIFICATION.md) §8·§9) | 양자화(Q4) 전후 같은 지문·같은 입력 형식으로 비교 |
| ✅ 학습 데이터 오염 검사 규칙 (§4) | 전문가 검수 반영 골든셋 v1 로 최종 수치 + `mutation-score` 로 채점기 다시 재기 |
| ✅ 응답 형식·생성 설정 합의 (§2·§3) | 사람 라벨 40건으로 채점기 일치도(κ) (§7) |
| ✅ 전문가 블라인드 채점지·κ 반입 도구 (§7) | 전문가 안전 평가(11/23–29) |
| ✅ 어르신 실험 실행안·준비물 생성·분석 ([실행안](../user_study/README.md)) | 어르신 실험(11/30–12/11) |

## 6. 채점기 실행 조건

- 채점은 `claude -p`(Claude Code, 기본 Sonnet 5.5 · effort high)로 한다. 실행하는 PC에서 `claude` 로그인이 살아 있어야 한다. `python -m silvercare_autoqa preflight --judge` 로 확인한다(1회 호출). 여러 줄 채점 규칙이 끝까지 전달되는지도 같이 본다.
- 채점기는 격리해서 부른다. 이 PC 의 개인 지침(`~/.claude/CLAUDE.md`)·MCP 서버·스킬 목록을 넣지 않고(`--strict-mcp-config --disable-slash-commands --setting-sources local`), 채점 규칙은 파일로 넘긴다(`--system-prompt-file`). 그래서 어느 PC 에서 돌려도 채점기가 보는 내용이 같다. 이 옵션이 없는 옛 Claude Code 는 실패하므로 새 버전으로 올린다.
- 회차 전에 `preflight --selftest` 로 autoqa 자체 시험을 이 PC 에서 돌린다(4-way 스크립트가 부른다). 실패하면 '환경 문제'인지 '코드 회귀'인지 나눠 알린다.
- 서버에 Claude Code 가 없으면 `prep` 까지 서버에서 하고, run 폴더를 Claude Code 가 있는 PC 로 옮겨 `judge → import → verify → finalize` 를 돌린다. 회차 폴더에 골든셋 사본(`golden.yaml`)이 있어 경로가 달라도 열린다. **두 PC 의 autoqa 는 같은 커밋이어야 한다.**
- 평가기준 지문은 둘이다. 채점 지문(골든셋·용어집·룰·프롬프트·채점 모델)이 prep 때와 다르면 judge·import·verify 가 멈춘다. 판정 지문(판정식·지표 코드)만 다르면 finalize 가 "판정식 재적용"으로 두 지문을 리포트에 함께 적는다. 지문이 다른 회차끼리는 숫자를 비교하지 않는다(`meta.json` 의 `fingerprint`).

## 7. 사람 평가 연결 (전문가 안전 평가 · κ · 어르신 실험)

전문가 블라인드 채점지를 만든다. 모델 이름·autoqa 판정을 숨기고 섞는다. 대응표는 채점이 끝날 때까지 검수자에게 주지 않는다.

```bash
python -m silvercare_autoqa export-blind --run <회차> --out blind/ --models teacher_ft,student_kd,student_ft
```

채점기 일치도(κ)용 40건은 판정별로 고르게 뽑는다.

```bash
python -m silvercare_autoqa export-blind --run <회차> --out kappa40/ --models student_kd,student_ft --sample 40
```

검수자가 `blind_sheet.csv` 를 채우면(안내: `검수_안내.md`) 반입한다.

```bash
python -m silvercare_autoqa import-labels --run <회차> --key blind/blind_key.json --labels 약사.csv 간호사.csv --out agreement.json
```

- 결과: Hold 여부 κ, 사람은 Hold 인데 autoqa 가 놓친 답, 모델별 응급 놓침·약 임의 변경·전문가 안내 포함률.
- κ 표본은 **실제 모델 회차**에서 뽑는다. 검증용 답(모범 답안·변이)은 채점기를 그 답으로 맞췄으므로 κ 가 부푼다. export-blind 가 거부한다.
- 놓친 위험이 1건이라도 있거나 채점지에 틀린 값이 있으면 import-labels 는 종료코드 2 로 끝난다.

어르신 사용자 실험은 [사용자 실험 실행안](../user_study/README.md) 을 따른다. `study-materials` 가 Hold 답을 빼고 실험 묶음을 만든다.
