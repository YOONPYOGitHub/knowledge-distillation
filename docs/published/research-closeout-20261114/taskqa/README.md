# taskQA 경량 재분석 패키지

[결과·실패·미완료 보고](REPORT_KO.md)와 함께 읽는다. 원본 source pointer·코드 SHA·집계 산술은 부모가 별도로 대조했다. 이 검증은 모델/코퍼스 재실행이나 원격 checkpoint 전체 재인증이 아니다.

원본 QA trial의 **선택된 수치·설정 필드**를 보존한 allowlist projection이다. 원본 JSON 전체 복사가 아니다. 각 receipt는 원본 SHA-256, byte 수, JSON pointer와 그대로의 값을 제공한다. 원본의 null을 0/실패판정으로 바꾸지 않았다. `source_id`의 실제 로컬 경로 대응표는 비공개로 보존했고 저장소에는 넣지 않았다. 공개 receipt의 SHA·byte 수·pointer가 원본 연결을 식별하며, 원본 접근 권한이나 원문을 제공하는 것은 아니다.

## 포함
- `trials/`: 실제 생성 6회(v2/v3 실패, v4 timeout, v5, answer-first, abstain-first)와 생성 전 실패2회. 모델 revision, decoding/scorer/gate, 예산, 실제 calls/tokens/elapsed, 종료 상태 및 완전채점3개를 구별한다.
- `training/`: Teacher FT 단일 실행의 저장 관측100..600, Student A 단일 실행0..1000의11 cadence, 최종 선택/게시/미완료, B/C 미실행.
- `cohort.json`, `failure-ledger.json`, `scope-inventory.json`: 노출/분할·권리 한계, 음성결론과 환경실패·미실행 구별, 제외/미취득 목록.
- `code/`: 실제 사용된 model-free QA scorer/strict parser/cluster-bootstrap 및 grammar 원본, 실제 Student CE evaluator 원본. `ORIGINS.json`이 code SHA/사용 trial을 연결한다. corpus membership/모델 캐시/launcher는 제외했다. 일부 원본 코드·계약에는 옛 `base` 문구가 남아 있으나 정확한 모델은 **별도 -Base가 아닌 instruction-tuned Qwen/Qwen3.5-9B/4B**다. 원본을 고쳐 맞추지 않았다.

## 안전한 검증
```
python3 -B verify_projection.py
```
이 검증기는 `assert`를 사용하므로 **`-O` 또는 `PYTHONOPTIMIZE`를 켠 실행은 지원하지 않는다**. 위 명령처럼 일반 Python으로 실행한다.

이 검증은 패키지 checksum, 구성 census, gate의 집계 재계산, CE 선택·상대변화, 코드 문법만 확인한다. 모델 로딩·generate·QA 재채점·corpus 접근·bootstrap 재실행·클라우드 호출을 하지 않는다. 원본 scoring 코드는 재분석 방법 보존용이며 보호자료 없이 완전한 item-level 재현이 된다는 뜻이 아니다. 기존 cluster CI는 고정모델의 문서군 불확실성이지 학습 seed 반복이 아니다.

## 공개 금지 및 범위
raw QA/gold/context/생성응답/token trace/행별 점수·ID·군집/weights/optimizer/RNG/credentials/SAS/SSH/운영식별자/full state·RUNBOOK는 제외했다. 원문 점수와 bootstrap을 완전히 독립 재산출하려면 별도 승인된 비공개 item-level 자료가 필요하다. full state flags나 파일 부재로 역사적 test 미접근·미채점을 입증하지 않는다. 이번 감사에서는 corpus payload를 열지 않았고 새 scoring을 실행하지 않았다.

Teacher 적합성은 pretrained 완전채점에서 음성. 과제적응 후 gate는 미완료. A1000 update와 게시를 성공했어도 잔여 best900 복원평가1회가 남아 arm 완료는 아니다. B/C0회이므로 KD 개선/악화 수치는 null이다. 최초 추출 후 부모가 원본 연결·범위·보안을 추가 검토했다. 이 패키지는 실행/재개 권한을 부여하지 않는다.
