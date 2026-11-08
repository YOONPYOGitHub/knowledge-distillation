# autoqa 작업 지침 (사람·AI 공통)

- 이 폴더의 문서와 autoqa 가 만드는 리포트·안내 문구는 `docs/autoqa/WRITING_GUIDE.md` 를 따른다. 기준은 하나다 — 처음 읽는 팀원이 되묻지 않는가.
- 고친 뒤 `python -m pytest silvercare_autoqa/tests -q` 를 돌린다. 문서 기준 검사(`test_docs.py`)가 여기 들어 있다.
- 판정 규칙·안전 Gate·평가기준을 바꾸면 `DESIGN.md`·`INTEGRATION.md` 의 해당 절과 `AUDIT_LOG.md` 를 같은 커밋에서 고친다.
- 공개 레포다. 회사 이름·사내 정보, AI Hub 원문(반출 금지)을 넣지 않는다.
- 이 파일과 짝 파일(`CLAUDE.md` ↔ `AGENTS.md`)은 내용이 같아야 한다 (시험이 확인한다).
