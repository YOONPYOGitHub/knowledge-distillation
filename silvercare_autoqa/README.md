# silvercare_autoqa — 실버케어 지식증류 모델 자동검수

어르신 질문 50문항(골든셋)으로 Teacher / Student KD / Student FT / Student Base 답을 채점한다. 안전 Gate 6종이 0건인지, Scorecard 7종이 목표에 닿는지 보고 모델마다 **Pass / Fix / Hold**를 낸다.

- 설계 = [docs/autoqa/DESIGN.md](../docs/autoqa/DESIGN.md)
- 골든셋 데이터 카드 = [docs/autoqa/GOLDEN_DATA_CARD.md](../docs/autoqa/GOLDEN_DATA_CARD.md)
- 실효성 검증 결과 = [docs/autoqa/VERIFICATION.md](../docs/autoqa/VERIFICATION.md)

## 준비

- Python 3.10 이상, `pyyaml`(레포 기본 의존성에 있음). 응답 생성을 HTTP로 받으면 `requests`도 쓴다.
- 채점은 이 PC의 Claude Code를 쓴다(`claude -p`, 기본 `claude-sonnet-5-5` · effort `high`). 먼저 터미널에서 `claude`를 한 번 실행해 로그인 상태를 확인한다. 다른 경로의 `claude`를 쓰려면 환경변수 `SILVERCARE_CLAUDE_BIN`을 정한다.

## 자주 쓰는 명령

```bash
python -m silvercare_autoqa validate                       # 골든셋 검증 + 모범 답안 자체 점검
python -m silvercare_autoqa guardrail --heldout silvercare_autoqa/data/golden/guardrail_heldout_v0.json   # 입력 가드레일 감지율

# 팀 UI 백엔드에서 4-way 답 받기 → 채점 → 리포트
python -m silvercare_autoqa generate --source kd-ui --run-id <run_id> --out responses.jsonl
python -m silvercare_autoqa run --responses responses.jsonl --run runs/20261120_kanana_v5 \
  --teacher teacher_ft --pairs student_kd:student_ft

# Claude Code 가 없는 곳: 요청 파일만 만들고, 다른 실행기가 results/ 를 채운 뒤 이어서
python -m silvercare_autoqa prep --responses responses.jsonl --run runs/x
python -m silvercare_autoqa import --run runs/x
python -m silvercare_autoqa finalize --run runs/x --teacher teacher_ft --pairs student_kd:student_ft

python -m pytest silvercare_autoqa/tests -q                # 단위·종단간 시험 (네트워크·LLM 없음)
```

응답 파일(JSONL) 한 줄 = `{"model": "student_kd", "item_id": "SC-E01", "response": "…"}`.

## 산출물 (회차 폴더)

| 파일 | 내용 |
|---|---|
| `report.md` | 모델별 최종 판정, 안전 Gate 표, Scorecard 표, 고칠 것(고칠 순서) |
| `report.html` | 문항마다 펼쳐 보기 (발화·응답·지적·근거·고치는 방법) |
| `defects.csv` | 사람 검토용. '사람 판단·사람 의견' 열에 이견을 적는다 |
| `metrics.json` | 모든 수치와 회차 유효성 |
| `meta.json` | 평가기준 지문 — 지문이 다른 회차끼리는 비교하지 않는다 |

## 고칠 때 지킬 것

- 판정식은 `verdict.verdict_of` 하나다. 다른 곳에 판정 규칙을 복제하지 않는다.
- 안전 지적은 확신도로 낮추지 않는다. 반박 검증(`verify`)만 뒤집을 수 있다.
- 용어집(`data/lexicons/*.yaml`)에 패턴을 넣을 때는 "확정(confirm)"과 "후보(hybrid)"를 구분한다. 부정문("~하지 마세요")에 걸리는 패턴은 후보로만 둔다.
- 골든셋·용어집·룰·프롬프트를 바꾸면 지문이 바뀐다. 이전 회차와 숫자를 비교하려면 같은 응답으로 다시 채점한다.
