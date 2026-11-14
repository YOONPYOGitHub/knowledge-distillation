# 공개 범위·검증·제외 목록

## 공개의 목적과 범위

성공한 최종 수치만 골라 올리지 않는다. 기존 `results/logs/`, `results/published/generic-kd-v1/`, 그림·보고서·학습/평가 helper는 유지하며, 이번 폴더에 **추가 실험별 수치·설정·선택·개발손실 이력·품질 기준·실패/종료 사유·실제 평가 구현**을 덧붙였다. 표와 CSV/JSON은 같은 실험의 대안 표현이며 서로 다른 반복 수로 세지 않는다.

## 포함/제외 inventory

| 분류 | 처리 | 재사용 범위 |
|---|---|---|
| 최종 집계·모델/revision·실험 설정·선택·종료/실패 | 공개 | 결과 재계산·조건/대조군 검토 |
| trial별 scalar/validation history·CSV 곡선 | 공개 가능한 필드 projection | 곡선/예산/선택 재분석. partial snapshot은 그대로 명시 |
| QA gate/CI·coverage·총계·평가 구현 | 공개 | 방법·집계 재계산. 원문 없는 item-level 재채점/CI 신규 계산은 불가 |
| 비의료 보조 대화 A/B/C·품질검토 총계·종료/timing metadata | 공개 | 같은 cohort 안에서 제한된 비교 |
| 원문 QA/gold/context·생성답변·prepared token/행별 점수·membership | 비공개 보존 | 원문 권리·개인정보·보호평가 경계 유지. 가벼운 파일이어도 자동 공개하지 않음 |
| 모델/adapter·optimizer/RNG·HF cache·코퍼스·대용량 archive | Git 제외·원본 보존 | 원격/로컬 보존과 별도 접근 권한. Git만으로 전체 재학습 재현 보장 안 함 |
| `.env`·토큰·비밀번호·키·쿠키·서명 URL·SSH/접속자료 | 제외 | 값·부분문자열·자격증명 fingerprint도 공개하지 않음 |
| full STATE/RUNBOOK·운영로그·writer/권한 설정·사적 연락처 | 그대로 공개하지 않고 필요한 과학값만 추출 | 현재 자동실행 권한/원장과 역사적 과학 기록을 분리 |
| 불완전/미취득/파싱 실패 근거 | inventory에 유지 | missing을 성공/0점/없는 실험으로 바꾸지 않음 |

- 기본 연구 전체 `results/research/`의 파일별 판정: [generic/research-inventory.json](generic/research-inventory.json).
- taskQA 범위/제외: [taskqa/scope-inventory.json](taskqa/scope-inventory.json), [실패 ledger](taskqa/failure-ledger.json).
- 서비스 근거: [service-results.json](evidence/service-results.json)의 각 원본 SHA/byte 수/pointer, [환경 기록](evidence/environment-results.json)의 원문 metadata projection/줄 excerpt.
- 논리 경로·source alias·SHA가 있다고 원본이 저장소에 포함되거나 공개 다운로드가 가능하다는 뜻은 아니다. 공개 파일은 이 폴더 manifest로 구분한다.

## 실제 검증 방법과 한계

### 원본에서의 추출 검증

부모가 metadata를 독립적으로 읽어 다음을 대조했다.

- generic source의 decode-chain/JSON pointer **13,921개 leaf occurrence**와 error token46개, trial76건, CSV216행.
- QA source SHA/pointer projection **385개** 및 원본 평가코드6개 bytes/SHA.
- service 원본 source SHA/pointer와 CE→PPL 및 A/B/C 상대변화. 환경 문서 excerpt는 원본 지정 줄과 대조.

이는 **보존된 metadata의 정확한 추출과 산술 검증**이다. 현재 클라우드 payload 전체, 가중치, 원문 분할·안전성, 과거 모든 비접근을 새로 인증하지 않는다. 원문 corpus/QA·모델을 열거나 GPU 실행·추론·새 채점을 하지 않았다.

원본 파일 SHA와 그 안에 기록된 과거 원격 report SHA는 다를 수 있다. generic trial에는 `source_sha256`(현재 retained compact bytes)와 `historical_report_sha256`(compact에 기록된 과거 report)을 구별했다. 파일의 SHA를 payload 전체 인증이나 독립적인 외부 attestation으로 과장하지 않는다.

### 공개 체크아웃에서 할 수 있는 검증

저장소 루트에서 모델·원문 취득 없이:

```bash
python3 -B results/published/generic-kd-v1/verify_results.py
python3 -B results/published/generic-kd-v1/test_verify_results.py
python3 -B docs/published/research-closeout-20261114/taskqa/verify_projection.py
```

첫 검증기는 기존 최종24개/9쌍과 선택 metadata를 재계산한다. QA 검증기는 checksum/구성·gate3개 재계산·A/Teacher 이력·코드 문법을 검사하며 실제 평가코드를 import/실행해 모델을 로딩하지 않는다. 음성결과와 미실행 B/C null을 그대로 유지한다. 체크아웃 전체 suite나 새 GPU 재학습 PASS라는 뜻이 아니다.

상위 `MANIFEST.json`의 각 파일은 SHA-256과 byte 수로 다시 검사할 수 있다. `.gitattributes`에 이 패키지의 줄바꿈 자동변환을 끄는 규칙을 추가했다. manifest 자체는 순환 해시를 만들지 않기 위해 자신의 inventory에서 제외했다.

## 해석 원칙

1. 같은 Student·tokenizer·cohort·평가조건에서 CE↔KD를 비교한다. 서로 다른 계열의 token PPL을 품질순위로 쓰지 않는다.
2. Base는 추가 domain/task 학습 전 상태라는 비교 역할과 공식 `-Base` variant를 구별한다.
3. 기존 FT 출발의 추가학습, 일반 평문 CPT, KLUE QA, 비의료 보조 대화, 의료 QA를 합치지 않는다.
4. 성능 악화의 확인과 그 원인 입증은 별개다. 검증되지 않은 FLA/JIT·Teacher quality·infrastructure 가설은 가설로 둔다.
5. 검증·실행·출판·서비스 승인은 다르다. CPU fixture/복구 시험·테스트 수를 실제 모델 update/임상 효능으로 세지 않는다.
6. 이 출판은 사용자 요청의 결과 보존 작업이며 중지한 cron·VM·작업공간을 재개하지 않는다.
