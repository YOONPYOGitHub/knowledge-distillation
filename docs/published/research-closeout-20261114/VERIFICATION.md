# 출판 검증 기록

이 기록은 문서·metadata 공개 묶음의 검증이다. 실제 모델을 재학습·추론·QA 재채점하거나 클라우드를 기동하지 않았다. “PASS”는 아래 각 범위에 한정한다.

## 원본 metadata와 대조

| 검사 | 실제 결과 |
|---|---|
| generic scientific leaf | 원본 JSON/decode-chain/pointer 13,921 occurrence 일치 |
| generic trial | Student76건의 설정/identity/초기·최선 지표·종료/선택 값 및 별도 frozen seed 일치 |
| generic CSV | 원래 row order·선택 열 216행 일치 |
| generic 오류 | 안전한 error token46 occurrence가 해당 원본 문자열에 존재. 독립 실패 incident46건이라는 뜻 아님 |
| QA projection | 원본 SHA와 pointer385개 대조 PASS |
| QA code | 평가/grammar/evaluator6개 원본 bytes/SHA 일치 |
| service | 원본 metadata SHA/pointer108개 일치, PPL=exp(CE) 및 A/B/C 상대효과 재계산 |
| service 후속 보완 | 원본26개 source항목의 pointer373개 추가 대조 후 private input경로·운영 next_action은 공개에서 제외. source항목 수는 고유 파일 수와 다름 |
| environment | metadata5개 원본 SHA·허용필드/원문 지정 줄 excerpt 일치 |

## 비공개 원본이 없는 Git index export에서 실행

tracked/staged 파일만 내보낸 별도 디렉터리에서 다음을 실제 실행했다.

1. 기존 generic `verify_results.py`: 결과24개 / CE-KD9쌍 / selected metadata21개 확인.
2. 기존 `test_verify_results.py`: **4 tests OK**.
3. taskQA `verify_projection.py`: trial문서8개 / 실제생성6개 / 시작calls1,124 / Teacher gate3개 재계산 / Student history11점 / Teacher history6점 확인.
4. 공개 manifest의 각 파일 SHA-256·byte 수 및 Markdown 상대 링크 대상을 검사했다.

이것은 원문·가중치 없이 가능한 검증이다. 원본 source pointer의 신규 대조는 위 로컬 보존 metadata가 있을 때만 가능하다. 파일 SHA는 클라우드 payload 인증이나 임상 안전성 검증이 아니다.

## 음성 fixture

공개 export의 **새 복사본/검증용 파일**에만 오류를 주입했다. 생산 학습 source·원래 결과는 바꾸지 않았다.

- `qa-order-answer-first`의 보존 Teacher gate를 false→true로 바꾸고 해당 파일 checksum도 새로 맞춤: **검증기 exit1**. 단순 checksum을 넘어 집계 gate 재계산으로 거짓 성공을 거부함.
- 원래 metadata 값으로 복원한 뒤 파일 byte를 변조하고 manifest는 원형 유지: **검증기 exit1**. byte/SHA 변조를 거부함.

fixture 실패 결과를 원래 실험의 실패로 계수하지 않는다.

## 보안·범위

공개 후보만 대상으로 개인 home 경로, subscription 경로, credential literal, private-key header, token/Bearer/SAS URL 패턴을 검사했다. 자동 패턴 검사에서 비밀이 안 나왔다는 사실은 완전한 보안/개인정보 인증이 아니므로 원문 QA·운영로그·full STATE 자체는 제외하고 구조·출처도 검토했다. 원본 state·writer lock·cron/비용 자원은 이 출판의 변경 대상이 아니다.
