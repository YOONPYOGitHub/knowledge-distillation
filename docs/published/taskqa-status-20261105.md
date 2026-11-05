# 한국어 문답 기반 후속 KD 연구 — 중간 공개 기록

**기준:** 2026-11-05. 완료된 기초 3계열 연구와 구분한 중간 결과다. 아래 QA 결과는 개발집합 조건부 평가이며, 진행 중 Teacher 적응이나 A/B/C 학습의 최종 성능표가 아니다. 서버 운영 상태는 11/05 21:58 KST 인계 기록을 반영한 저장 관측이다.

## 1. 기초 연구와 다른 연구 질문

기초 연구는 한국어 평문에서 동일 Student의 **정답 토큰 CE 대 CE+Teacher KL**을 비교했다. [기초 완료 보고](generic-kd-v1/README.md).

후속 연구는 근거 문맥이 있는 한국어 질문에 정확히 답하고, 문맥으로 답할 수 없으면 적절히 거절하는지를 다룬다. 평문 PPL 개선만으로 문답·의료안전·실버케어 효용을 주장하지 않는다. 다음 세 대조군을 분리한다.

| 대조군 | 학습 표적·손실 | 식별하려는 효과 |
|---|---|---|
| A | 독립 gold 정답, response-only CE | 일반 과제 적응 기준선 |
| B | 우위·품질 검수를 통과한 Teacher 응답, response-only CE | Teacher 응답 데이터로 바꾼 효과: A↔B |
| C | B와 **같은 응답·prefix**에 Teacher KL 추가 | KL의 추가 효과: B↔C |

A↔C만 비교하면 응답 데이터 변경과 KL 효과가 섞인다. 각 Student는 같은 초기화·자료·마스크·학습량/선택 기준을 사용한다. Teacher가 더 큰 모델이라는 이유만으로 유효한 정답 공급자로 가정하지 않는다. 추가 on-policy/새 손실/양자화 회복은 필요성이 확인될 때 별도 계약으로 다룬다.

## 2. 데이터와 공개 범위

- 공식 소스: [KLUE / MRC](https://huggingface.co/datasets/klue/klue), `klue/klue`, config `mrc`, **공식 train**.
- 고정 revision: `349481ec73fff722f88e0453ca05c77a447d967c`.
- 공식 train 17,554행을 대상으로 정제·문서 proxy·문맥/근접중복 군집을 구성한 뒤 내부 1,000문항을 채택했다. 선택한 1,000행만 노출됐다고 주장하지 않는다.
- 내부 분할: train **794**, dev **124**, sealed test **82**. 이것은 공식 validation/test 성적표가 아니다.
- 원문 source/title 기반 문서 proxy와 lexical near-cluster를 사용한다. 내부 dev124문항은 문서군90개이며 answerable93 / impossible31이다.
- `is_impossible` 문항의 비어 있지 않은 upstream answer span은 plausible distractor일 수 있어 그대로 정답으로 쓰지 않는다. 원형 span/offset을 보존하고 응답가능성 의미를 구분한다.
- 데이터 카드의 CC-BY-SA-4.0 표시 외에 Wikipedia 원자료·Hankyung/Acrofan의 KLUE-MRC/기계학습 목적 권리 조건이 있다. **원문 재배포·일반 뉴스 재사용·상업 서비스 지식으로의 배포 권리는 별도 검토 대상**이다.

이 커밋에는 raw/train/dev/sealed 문항·gold·모델 응답·token trace를 넣지 않는다. 위 링크는 **공식 원천 접근 경로**이며 현재 내부 분할을 동일하게 재생성하는 완전한 배포 패키지라는 뜻이 아니다. 정확한 cohort로 협업할 때는 담당자에게 승인된 metadata/자료 접근을 요청하고, 이미 노출된 dev와 보호된 test를 분리한다.

## 3. 완료된 QA 개발평가

### 3.1 사전학습 Qwen Teacher / Student

아래는 동결된 출력·채점 계약으로 **각124문항 전체를 채점한 기존 v5 결과**다. EM은 정확일치, F1은 정답 중첩 지표다.

| 모델 | 전체 EM | answerable F1 | impossible 거짓답변 | answerable 과잉거절 |
|---|---:|---:|---:|---:|
| Teacher 9B | 79/124 (63.71%) | 83.14% | 21/31 (67.74%) | 1/93 (1.08%) |
| Student 4B | 82/124 (66.13%) | 80.56% | 16/31 (51.61%) | 3/93 (3.23%) |

Teacher는 answerable F1의 점추정은 높았지만 전체 EM과 불가능 문항 거짓답변이 불리했다. 동결된 **Teacher 적합성 gate는 통과하지 못했다**. 큰 모델의 출력을 검수 없이 증류 표적으로 채택하지 않는 근거다. 이 점수는 QA 과제 적응이나 Student KD 학습 후 점수가 아니다.

### 3.2 답변/거절 출력순서 대조

| 출력순서 | 모델 | 전체 EM | answerable F1 | impossible 거짓답변 | answerable 과잉거절 |
|---|---|---:|---:|---:|---:|
| answer-first | Teacher | 79/124 | 83.14% | 21/31 | 1/93 |
| answer-first | Student | 82/124 | 80.56% | 16/31 | 3/93 |
| abstain-first | Teacher | 49/124 | 22.39% | 0/31 | 66/93 |
| abstain-first | Student | 56/124 | 31.31% | 1/31 | 60/93 |

거짓답변 감소와 동시에 답할 수 있는 질문의 과잉거절이 크게 증가했다. 두 순서 모두 Teacher 우위 gate=false로 해당 가설을 음성결론으로 닫았다. answer-first와 기존 v5의 일치 집계를 별도 독립 학습 반복으로 세지 않는다. grammar 제약은 새로운 decoding 개입이며, 출력 형식 성공률을 정답률로 대체하거나 malformed 응답을 사후 수정하지 않는다.

## 4. 아직 완료되지 않은 것

11/05 00:34:48 KST의 마지막 인증된 모델 관측은 **Teacher latest600 / best500 step**이다. 동일 Teacher 실행1회가 중단됐고 완료0회이며, 이 후속 과제의 **Student A/B/C 학습은 각각0회**다. 복구·직렬화·합성 optimizer 시험은 실제 과제학습 건수에 더하지 않는다.

11/05 21:58 KST 저장 인계에서는 VM 전원은 running이지만 서버 내부 진단 결과·현재 GPU·체크포인트 인증이 미완료였다. **현재 학습이 계속 실행 중이라는 주장이 아니다.** 복구 후 실제 cursor·남은 예산·Teacher 평가를 인증한 다음 제한된 Teacher-only 적응을 마치고, 우위/답변 검수 gate에 따라 후속 A/B/C를 판단한다.

따라서 현재 공개할 수 있는 결론은 다음과 같다.

- 완료: 기초 3계열 평문 비교, 사전학습 QA 적합성 평가, 두 출력순서 대조의 음성결론.
- 미완료: 과제 적응 후 Teacher 성능, A/B/C 비교와 그 KD 효과, 추가 seed 재현성.
- 미검증: 의료안전, 실어르신 효용, 실제 기기 양자화·RAM/지연/발열, 상용 서비스 준비.

## 5. 다른 4-way 개발자가 참고할 부분

1. 먼저 [공동 재실험 가이드](../FOUR_WAY_REPRODUCTION_GUIDE.md)에 따라 데이터·입력 처리·손실 집계·validation 선택·학습량/탐색 예산을 합의한다.
2. 일반 평문 재실험은 [기초 완료 결과](generic-kd-v1/README.md)와 공통 방법을 참고하되, 이 QA pilot을 이미 완료된 KD 검증 코드로 취급하지 않는다.
3. QA를 새로 한다면 독립 정답 학습과 Teacher 응답 학습, KL 추가를 분리하고 답변가능/불가능·거짓답변/과잉거절·형식실패를 함께 기록한다.
4. 같은 test를 사용한 동결 설정의 재현과 test 결과를 참고한 새 튜닝을 구별한다. 새 방법 개발 뒤 독립 일반화 주장이 필요하면 별도의 공동 heldout 정책을 먼저 합의한다.

## 6. 출처와 소프트웨어 범위

이 공개 요약은 로컬 `parallel/report_draft/REPORT_DRAFT.md`(2026-11-03)의 완료 집계·출처 ledger, `state.json`의 데이터 metadata, `phase2e/recovery_followup_20261105T2201/HANDOFF.md`(실제 관측21:58)의 운영 경계에 근거한다. 원문 SHA와 포함/제외 범위는 옆 `PUBLICATION.md`에 기록한다. 원본·과거 음성 결과를 덮어쓰거나 재채점하지 않았다. 독립 문서 검토가 새 모델 실행·전체 원본 재인증을 뜻하지 않는다.

후속 QA 실행 패키지는 별도 작업공간에서 개발 중이며 **이 저장소 공개분은 그 전체 실행 코드를 포함하지 않는다.** 이 커밋의 `src/research_*.py`와 학습/평가 CLI는 generic 연구용이며, 아래 문서만으로 production QA 학습·의료서비스가 실행된다고 보장하지 않는다.
