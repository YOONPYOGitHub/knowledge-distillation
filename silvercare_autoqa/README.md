# silvercare_autoqa — 온담(로컬 LLM + RAG) 자동 채점(autoqa) 코드

## 한눈에

- 이 폴더는 autoqa 를 직접 돌리거나 고치는 사람용이다. 팀원은 먼저 [연결 안내](../docs/autoqa/INTEGRATION.md) 를 본다. 역할별 할 일은 [README](../README.md#역할별-할-일) 에 있다.
- 골든셋으로 온담·ChatGPT·검색 끈 로컬 모델의 답을 각각 채점하고 직접 비교한다. 답마다, 모델마다 **Pass / Fix / Hold** 를 낸다.
- 2026-11-14 팀 방향이 지식증류에서 온디바이스 LLM + RAG 로 바뀌었다. 지식증류 때 수집기(`kd-ui`)와 4-way 스크립트는 남겨 두었다.
- 채점은 채점하는 PC 의 Claude Code(`claude -p`, 기본 `claude-sonnet-5-5` · effort `high`)로 한다. 시험(`pytest`)은 네트워크·LLM 없이 돈다.
- 명령 전체는 아래 「명령」 표가 정본이다. 판정 규칙은 [DESIGN.md](../docs/autoqa/DESIGN.md), 고친 기록은 [AUDIT_LOG.md](../docs/autoqa/AUDIT_LOG.md) 에 있다.

## 준비

- Python 3.10 이상과 `pyyaml` 이 필요하다. 답을 HTTP 로 받으면 `requests` 도 쓴다.
- 채점하는 PC 에 Claude Code 가 설치·로그인돼 있어야 한다. 확인: `python -m silvercare_autoqa preflight --judge` (Sonnet 1회 호출).
- 다른 경로의 `claude` 를 쓰려면 환경변수 `SILVERCARE_CLAUDE_BIN` 을 정한다.

## 명령

모두 `python -m silvercare_autoqa <명령>` 으로 부른다. 옵션은 `<명령> --help` 로 본다.

| 명령 | 하는 일 | LLM 호출 |
|---|---|---|
| `validate` | 골든셋 형식 검사 | 없음 |
| `guardrail` | autoqa 응급 키워드 목록의 감지율 (`--heldout` 이면 처음 보는 발화 60개) | 없음 |
| `preflight` | 실행 전 점검 — 골든셋·팀 UI 백엔드·자체 시험(`--selftest`)·로그인(`--judge`) | `--judge` 면 1회 |
| `generate` | 답 받기 — 온담 API·OpenAI 공식 API·손으로 받은 서식·OpenAI 호환(Ollama)·HF | 채점 LLM 은 안 씀 |
| `sheet-template` | ChatGPT 앱 답을 손으로 받을 빈 서식 | 없음 |
| `guardrail-app` | 응급 발화 시험셋 60개를 온담에 넣어 응급 고정 안내로 가는지 | 없음 (온담 호출 60회) |
| `corpus-coverage` | 골든셋 문항의 근거가 RAG 지식 자료에 있는지 핵심어로 센다 | 없음 |
| `check-responses` | 응답 파일 점검 — 빠진 문항·빈 답·프롬프트 섞임·생성 오류 | 없음 |
| `prep` | 결정 룰을 돌리고 채점 요청 파일을 만든다 | 없음 |
| `judge` | 채점 | 빈 답을 뺀 답 수만큼 |
| `import` | 채점 결과를 검사하고 반박 검증 요청을 만든다 | 없음 |
| `verify` | 반박 검증 | Hold 가 될 지적이 있는 답 수만큼 |
| `finalize` | 판정·지표·리포트 | 없음 |
| `run` | prep → judge → import → verify → finalize 를 한 번에 | judge·verify 만큼 |
| `compare` | 두 시스템 블라인드 직접 비교 + 비열등 판정 | 공통 문항 × 2 (순서 둘) |
| `contamination` | 골든셋이 RAG 자료·학습 데이터에 섞였는지 | 없음 |
| `export-blind` | 약사·간호사용 블라인드 채점지 | 없음 |
| `import-labels` | 채운 채점지 → 사람과 autoqa 의 일치도(κ)·놓친 위험 | 없음 |
| `study-check` | 어르신 실험 문항 답에 Hold 가 없는지 | 없음 |
| `study-materials` | 어르신 실험 묶음 (카드·대본·기록지) | 없음 |
| `study-analyze` | 어르신 실험 결과 분석 | 없음 |
| `mutants` | 변이 시험용 결함 답 만들기 (미묘한 변이는 요청 파일만) | 없음 |
| `mutation-score` | 변이 시험 회차의 탐지율을 다시 낸다 | 없음 |

## 자주 하는 일

채점하는 PC 에서 실행 전 점검을 하려면 (골든셋·자체 시험·채점기 로그인, Sonnet 1회 호출):

```bash
python -m silvercare_autoqa preflight --selftest --judge
```

온담과 ChatGPT 를 각각 채점하고 직접 비교하려면 (환경변수는 스크립트 머리말):

```bash
ONDAM_URL=https://<온담 주소> COMPARE_MODEL=<OpenAI 모델 이름> bash scripts/autoqa_rag_compare.sh
```

응답 파일(JSONL)이 이미 있으면 (형식: [INTEGRATION.md](../docs/autoqa/INTEGRATION.md) §2):

```bash
python -m silvercare_autoqa run --responses ondam.jsonl chatgpt.jsonl --run runs/20261120_ondam --reference chatgpt_luna61 --pairs ondam_cloud:chatgpt_luna61
```

이어서 직접 비교와 비열등 판정을 하려면:

```bash
python -m silvercare_autoqa compare --run runs/20261120_ondam --a ondam_cloud --b chatgpt_luna61
```

다른 실행기로 채점하는 순서는 [INTEGRATION.md](../docs/autoqa/INTEGRATION.md) §6 에 있다. 순서를 바꾸면 반박 검증이 빠져 회차가 유효성 미달이 된다.

약사·간호사에게 보낼 블라인드 채점지를 만들려면:

```bash
python -m silvercare_autoqa export-blind --run runs/x --out blind/ --models ondam_cloud,chatgpt_luna61
```

어르신 실험 묶음을 만들려면:

```bash
python -m silvercare_autoqa study-materials --run runs/x --out study_bundle/
```

## 채점 회차 폴더에 생기는 것

| 파일 | 내용 |
|---|---|
| `report.md` | 첫머리 '한눈에' · 모델별 판정 · 안전 Gate · Scorecard · RAG 지표 · 고칠 것 |
| `report.html` | 같은 표 + 문항마다 펼쳐 보기 (발화·응답·지적·근거·고치는 방법) |
| `defects.csv` | 팀 안 검토용. '사람 판단·사람 의견' 열에 이견을 적는다. 판정이 보이므로 κ 에는 쓰지 않는다(κ 는 `export-blind`) |
| `metrics.json` | 모든 수치와 회차 유효성 |
| `meta.json` | 평가기준 지문 — 지문이 다른 회차끼리는 숫자를 비교하지 않는다 |
| `golden.yaml` | prep 때 쓴 골든셋 사본 — 다른 PC 에서도 같은 기준으로 연다 |
| `compare/<A>__vs__<B>/` | 직접 비교 — `report.md`(비열등 판정) · `compare.json` · 요청·결과 파일 |

## 고칠 때 지킬 것

- 판정은 `verdict.verdict_of` 한 함수만 내린다. 다른 곳에 판정 규칙을 복제하지 않는다.
- 안전 Gate 표는 `gates.py`, 결정 룰의 종류·형식 실패 여부는 `rules.RULES` 한 곳에만 적는다.
- 안전 지적은 확신이 낮다는 이유로 낮추지 않는다. 반박 검증(`verify`)만 근거 인용으로 뒤집을 수 있다.
- 규칙 단어 목록(`data/lexicons/*.yaml`)에 패턴을 넣을 때 확정(confirm)과 후보(hybrid)를 나눈다. 부정문에도 걸리는 패턴은 후보로만 둔다.
- 골든셋·규칙 단어 목록·결정 룰·프롬프트·자료 묶음표(`data/rag_sources.yaml`)를 바꾸면 채점 지문이 바뀐다.
- 온담 출처 표시값이 바뀌면 `data/rag_sources.yaml` 만 고친다. 묶음을 알아보지 못한 출처는 리포트 §3-2 에 경고가 뜬다.
- 지문이 바뀌면 예전 회차와 숫자를 비교할 수 없다. 비교하려면 같은 응답으로 새 회차를 만들어 다시 채점한다. 어느 파일이 어느 지문인지는 [DESIGN.md](../docs/autoqa/DESIGN.md) §6.
- 문서·리포트 문구는 [문서 쓰는 법](../docs/autoqa/WRITING_GUIDE.md) 을 따른다. 고친 뒤 `python -m pytest silvercare_autoqa/tests -q` 를 돌린다.
