# 한국어 KD·QA·실버케어 연구 결과 보존본

정리 기준: **2026-11-14** · 브랜치: `feature/azure-h100-kd-tuning`

**성공한 수치뿐 아니라 성능이 나빠진 결과·품질 기준 미달·실행 실패·미완료를 함께 보존한다.** 이 문서는 새 실험 제안서가 아니라 지금까지 남은 결과를 활용하기 위한 색인이다. 사용자 요청으로 자동 실행·복구·정기보고를 중지했으며, 과거 실행 안내를 재개 승인으로 해석하지 않는다.

## 먼저 읽을 결론

| 연구 범위 | 실제 확인한 결과 | 해석 한계/종료 판정 |
|---|---|---|
| 과거 소형 한국어 H100 튜닝 | CE best PPL `11.814911107554344` → 선택 KD `11.353113405876357` (**−3.908601%**) | 기존 FT Student 출발의 단일 seed·소규모 Wikipedia. 다른 초기설정의 부정 결과도 별도 보존 |
| 공통 한국어 평문 3계열 | Qwen 세 Student seed 모두 소폭 개선; Gemma·Liquid는 개선/악화 혼재 | 최종 test 결과. KD의 일반적·실용적 우월성, cross-tokenizer 품질 순위는 미입증 |
| KLUE 기반 taskQA | Teacher 사전학습 QA 적합성 gate 음성. Student A는 **1,000 update**·best900 게시까지 진행 | 선택 best의 최종 복원평가 미완료. B/C **미실행**, 따라서 QA KD 효과는 null |
| 비의료 보조 대화 41행 적응 | Teacher dev PPL `15.76762739041548`→`5.446563675300413`; Student `17.28518697873459`→`5.634106403608484` | 내부 dev11·유효201토큰. 목표 응답 품질/의료안전 승인과 별개 |
| 비의료 matched12 A/B/C | B→C dev PPL **−2.470751%**, A→C **+0.122265%** | KL은 같은 Teacher 응답 CE보다 개선했지만 정답 CE는 넘지 못함. 한 seed·작은 dev |
| 앱/복구 기능시험 | 과거 비공개 비의료 단일 요청 E2E, 한정 CPU checkpoint 경계 시험 등 성공 | 상시 서비스 전환·의료 효능·실제 저사양 기기 검증은 미완료 |

이 표의 각 행은 서로 다른 데이터·목적·학습/평가 조건이다. 수치를 합쳐 하나의 개선율이나 모델 순위로 만들지 않는다. 원인 가설과 확인된 오류도 구별한다.

## 1. 연구 결과를 실제로 활용할 파일

### 기본 한국어·과거 실험

- [기존 기본 3계열 최종보고](../generic-kd-v1/REPORT.md): 모델·revision·데이터·선택·제어·한계.
- [기존 최종 JSON 24개 / CE-KD 9쌍 / 검증기](../../../results/published/generic-kd-v1/README.md).
- [확장 실험별 패키지](generic/PACKAGE.md): **Student trial76건**의 수치·설정·종료/선택, **379개 metadata source**의 경량 기록, CSV곡선7개·216행, 파일1,385개 포함/제외 inventory. 새 최종 결과만 추린 요약이 아니다.
- [역사적 긍정·부정 결과와 최종 수치 감사](generic/REPORT_KO.md), [집계 근거](generic/evidence.json).
- [과거 H100 전체 평가 JSON](../../../results/logs/h100-tuning-20260920/test_results.json), [튜닝 CSV](../../../results/logs/h100-tuning-20260920/parameter_search.csv), [과거 보고서](../../../results/logs/h100-tuning-20260920/report.md). 이전 `results/logs/`·그림은 그대로 유지했다.

### KLUE 기반 taskQA

- [결과·실패·미완료 보고](taskqa/REPORT_KO.md).
- [재분석 패키지 안내와 검증](taskqa/README.md): 실제 생성6회, 생성 전 실패2회, 완전채점3개 조건의 수치·gate·CI, Teacher/A history, B/C 미실행.
- [trial별 수치·설정](taskqa/trials/), [Teacher FT](taskqa/training/teacher-FT.json), [Student A](taskqa/training/student-A.json), [미실행 B/C](taskqa/training/unexecuted-B-C.json).
- [원래 평가/strict parser/grammar 코드6개와 source SHA](taskqa/code/ORIGINS.json). 보호 원문 없이 완전한 item-level 재채점이 가능하다는 뜻은 아니다.
- [cohort](taskqa/cohort.json), [실패 ledger](taskqa/failure-ledger.json), [포함/제외 범위](taskqa/scope-inventory.json).

### 실버케어 보조 대화·서비스 경로

- [측정값·A/B/C·품질/생성/앱 한계](SILVERCARE.md).
- [원본 metadata SHA·JSON pointer와 그대로의 수치](evidence/service-results.json).
- [개발손실 곡선 CSV](evidence/service-learning-curves.csv), [A/B/C 상대효과 CSV](evidence/service-paired-effects.csv).
- [후속 정책 전후 검토24건·train응답 검토43건/12개 제한 수용·PEFT3/Azure 후보 실제 추론 근거](evidence/service-supplement.json).
- [환경·데이터접근·복구 경계 근거](evidence/environment-results.json).

## 2. 실패와 실험이 어려웠던 이유

[실패/미완료 분석](FAILURES.md)은 다음을 구별한다.

1. 유효한 성능 미개선/혼재: KD가 대조군을 항상 이기지 못함.
2. 지표는 개선했으나 응답 품질 gate 실패: 낮은 dev loss와 실제 목적에 맞는 답변은 다른 기준.
3. 생성/평가 계약 실패: strict parser, timeout·부분 coverage 등.
4. 환경/구현 차단: Spot 회수/할당 부족, loader/dtype/timeout, 후속 서비스 구성요소·종료·메모리/코드 동일성 검증 미완료.
5. 데이터 단계 차단: 파일 취득·QA 연결·의미/임상 수용·권리는 학습 실행과 별개.
6. 미완료 및 사용자 중지: 미실시 결과를 “개선율0%”로 쓰지 않음.

**왜 효과가 없었는지 인과적으로 입증하지 못한 경우에는 미확정이라고 썼다.** 인프라 장애가 있었다고 “실행했으면 더 좋았을 것”이라 추정하지 않는다.

## 3. 보존·검증·재사용 범위

- [포함/제외 및 검증 범위](PUBLICATION.md), [실행한 검증·음성 fixture](VERIFICATION.md)
- [파일별 bytes/SHA manifest](MANIFEST.json)
- 기존 [4-way 재실험 가이드](../../FOUR_WAY_REPRODUCTION_GUIDE.md), [기본 결과 검증기](../../../results/published/generic-kd-v1/verify_results.py)

모델/optimizer·대용량 코퍼스뿐 아니라 **원문 QA·생성문·개인정보·비밀/접속정보**도 제외했다. 대신 경량 지표·설정·선택·오류·종료 근거·평가 구현과 source SHA/pointer를 남겼다. 제외한 항목의 존재/범위와 이유를 inventory에 명시했으며 로컬 원본은 지우지 않았다.

**완전성의 범위:** 모든 실험을 새로 실행하거나 모든 원격 payload를 회수한 전수검증은 아니다. 보존 metadata에서 확인 가능한 경량 과학 결과를 폭넓게 공개하며, 없거나 불완전한 곡선·per-document loss·선택 복원평가 등은 누락/미완료로 표시했다. payload가 빠진 것을 숨기거나 해시만으로 접근 가능하다고 주장하지 않는다.
