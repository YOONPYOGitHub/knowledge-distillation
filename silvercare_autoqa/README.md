# silvercare_autoqa — 실버케어 지식증류 모델 자동검수

## 한눈에

- 어르신 질문 50문항(골든셋)으로 Teacher·Student KD·Student FT·Student Base 답을 채점한다.
- 안전 Gate 6종이 0건인지, Scorecard 7종이 목표에 닿는지 보고 모델마다 **Pass / Fix / Hold** 를 낸다.
- 채점은 이 PC 의 Claude Code(`claude -p`, Sonnet 5.5 · effort high)로 한다. 시험(`pytest`)은 네트워크·LLM 없이 돈다.
- 처음 보는 분은 [브랜치 첫 화면](../README.md) → [연결 안내](../docs/autoqa/INTEGRATION.md) 순서로 본다.

## 준비

- Python 3.10 이상과 `pyyaml` 이 필요하다. 답을 HTTP 로 받으면 `requests` 도 쓴다.
- 채점 전에 터미널에서 `claude` 를 한 번 실행해 로그인 상태를 확인한다.
- 다른 경로의 `claude` 를 쓰려면 환경변수 `SILVERCARE_CLAUDE_BIN` 을 정한다.

## 자주 쓰는 명령

실행 전 점검 (골든셋·자체 시험·채점기 로그인):

```bash
python -m silvercare_autoqa preflight --selftest --judge
```

팀 UI 백엔드에서 4-way 답을 받아 채점·리포트까지:

```bash
bash scripts/autoqa_4way.sh <run_id>
```

응답 파일(JSONL)이 이미 있으면:

```bash
python -m silvercare_autoqa run --responses responses.jsonl --run runs/20261120_kanana_v5 --teacher teacher_ft --pairs student_kd:student_ft
```

Claude Code 가 없는 서버에서는 요청 파일만 만들고, 회차 폴더를 옮겨 이어서 돌린다:

```bash
python -m silvercare_autoqa prep --responses responses.jsonl --run runs/x
python -m silvercare_autoqa judge --run runs/x
python -m silvercare_autoqa import --run runs/x
python -m silvercare_autoqa verify --run runs/x
python -m silvercare_autoqa finalize --run runs/x --teacher teacher_ft --pairs student_kd:student_ft
```

사람 평가 연결 (전문가 블라인드 채점·κ·어르신 실험):

```bash
python -m silvercare_autoqa export-blind --run runs/x --out blind/ --models teacher_ft,student_kd,student_ft
python -m silvercare_autoqa study-materials --run runs/x --out study_bundle/
```

명령 전체 목록은 `python -m silvercare_autoqa --help` 로 본다.

## 회차 폴더에 생기는 것

| 파일 | 내용 |
|---|---|
| `report.md` | 첫머리 '한눈에' · 모델별 판정 · 안전 Gate · Scorecard · 고칠 것 |
| `report.html` | 같은 표 + 문항마다 펼쳐 보기 (발화·응답·지적·근거·고치는 방법) |
| `defects.csv` | 팀 안 검토용. '사람 판단·사람 의견' 열에 이견을 적는다 |
| `metrics.json` | 모든 수치와 회차 유효성 |
| `meta.json` | 평가기준 지문 — 지문이 다른 회차끼리는 숫자를 비교하지 않는다 |
| `golden.yaml` | prep 때 쓴 골든셋 사본 — 다른 PC 에서도 같은 기준으로 연다 |

## 고칠 때 지킬 것

- 판정식은 `verdict.verdict_of` 하나다. 다른 곳에 판정 규칙을 복제하지 않는다.
- 안전 Gate 표는 `gates.py`, 결정 룰의 종류·형식 실패 여부는 `rules.RULES` 한 곳에만 적는다.
- 안전 지적은 확신이 낮다는 이유로 낮추지 않는다. 반박 검증(`verify`)만 근거 인용으로 뒤집을 수 있다.
- 용어집(`data/lexicons/*.yaml`)에 패턴을 넣을 때 확정(confirm)과 후보(hybrid)를 나눈다. 부정문에도 걸리는 패턴은 후보로만 둔다.
- 골든셋·용어집·룰·프롬프트를 바꾸면 평가기준 지문이 바뀐다. 예전 회차와 비교하려면 같은 응답으로 다시 채점한다.
- 문서·리포트 문구는 [문서 쓰는 법](../docs/autoqa/WRITING_GUIDE.md) 을 따른다. 고친 뒤 `python -m pytest silvercare_autoqa/tests -q` 를 돌린다.
