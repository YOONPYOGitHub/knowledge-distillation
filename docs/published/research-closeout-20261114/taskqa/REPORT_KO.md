# taskQA 연구 종료 감사 — 공개용 결과와 미완료 구분

## 결론

- **완료된 QA 성과는 사전학습 모델 적합성·출력순서 개발평가와 그 음성결론**이다. Teacher가 더 큰데도 우위 gate를 통과하지 못했고, 거절을 앞에 배치하면 거짓답변은 줄지만 답할 수 있는 질문의 과잉거절이 크게 증가했다.
- **과제적응의 실제 진척도 남긴다.** Teacher는 역사적 latest600 / best500, best dev CE **0.0934396254011231**. Student A gold-CE는 **1,000 update**와 latest1000/best900 원격게시까지 도달했고 best dev CE **0.08195323988509474**다. 다만 선택 best900 복원평가1회가 남아 최종 arm 완료는 아니다.
- B Teacher-response CE 및 C KD는 **미실행**이다. 그러므로 taskQA에서 KD가 CE를 이겼다거나 졌다는 효과 추정은 **null**이다. Teacher FT도 완료된 적응 후 생성 QA/gate 결과가 없다.
- 11/05 공개문서의 “A/B/C 각각0회”는 당시 snapshot으로는 보존하되 현재 종료요약에는 A의 후속 실제 진척을 반영해야 한다. 최신 보존근거는 11/08 pause와 원본 receipts이며, 과거 state의 running 값들을 현재상태로 재사용하지 않았다. 이 감사는 live cloud 상태를 확인하지 않았다. 연구 중지 지시를 유지하고 재개하지 않았다.

## 1. 정확한 모델·평가 설정

Teacher **Qwen/Qwen3.5-9B**, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` → Student **Qwen/Qwen3.5-4B**, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

별도 `-Base` checkpoint가 아니라 **instruction-tuned pretrained** 모델이다. “unchanged baseline/base”는 추가 과제학습 전이라는 뜻과 공식 Base variant를 구별해야 한다. 초기 임시 contract/원본 코드의 남아 있는 `base` 문구보다 동결 execution과 실제 후속 contract의 모델 identity를 우선했다.

공통 BF16, adapter 없음(사전학습 평가), 공식 chat template·non-thinking, greedy/beam1, input8192/output256, truncation 금지. grammar 출력은 decoding 개입이며 정답성·의료안전 검증이 아니다. EM/F1은 로컬 NFKC/casefold·Unicode punctuation/symbol 정규화·whitespace multiset F1이며 **공식 KLUE 점수와 같다고 부르지 않는다**. 원본 설정과 gate는 각 trial JSON의 `configurations`에 보존했다.

근거: 공개 `trials/qa-order-answer-first.json` 등 각 trial의 `configurations`에 모델/decoding/gate/scorer 및 원본 source_id·SHA·정확한 JSON pointer를 보존했다.

## 2. 완전채점 개발결과

동일 dev124 / 문서군90, answerable93 / impossible31. 아래 값은 원본 score-only 수치에서 합산하고 원래 summary와 대조했다. 문항·gold·응답을 새로 읽거나 재채점하지 않았다.

| 출력순서 | 모델 | 전체 EM | answerable F1 | impossible 거짓답변 | answerable 과잉거절 |
|---|---|---:|---:|---:|---:|
| answer-first / v5 | Teacher | 79/124 (63.71%) | 83.14% | 21/31 | 1/93 |
| answer-first / v5 | Student | 82/124 (66.13%) | 80.56% | 16/31 | 3/93 |
| abstain-first | Teacher | 49/124 (39.52%) | 22.39% | 0/31 | 66/93 |
| abstain-first | Student | 56/124 (45.16%) | 31.31% | 1/31 | 60/93 |

**Teacher gate:** answerable F1차이≥0.05, 문서군 bootstrap95%구간 하한>0, 전체 EM차이≥0, answerability차이≥−0.02, impossible 거짓답변률은 Student보다 나쁘지 않아야 한다. 사전동결 bootstrap2,000회/seed20261031.

answer-first에서 Teacher−Student F1차이는 **0.0257609773738805**, 저장95%구간 **[−0.04470031686526532, 0.09683023247379681]**. 점차이는 요구5pp에 미달하고 구간하한·EM·answerability·거짓답변 조건도 충족하지 못했다. abstain-first 구간은 **[−0.18713738368910787, 0.0033234126984126914]**. 두 순서 모두 gate=false.

이는 고정모델·해당 dev·출력조건의 음성결론이다. 문서군 구간은 training seed 재현성이 아니다. v5와 answer-first의 동일 집계를 독립 학습 반복으로 더하지 않는다. full paired generation은3회지만 독립적인 학습seed 효과 검증은 아니다. 적응 후 Teacher의 우위와 응답검수는 아직 충족되지 않았다.

근거: `score_answer-first`, `score_abstain-first`, `v5`, JSON pointers `/summary`, `/cluster_uncertainty`, `/teacher_advantage_gate`; 합계는 `/per_question`의 숫자 필드만 사용했고 공개 패키지에는 행별 점수·ID·군집을 넣지 않았다.

## 3. 완료·실패 trial 전부의 경량 census

| trial | 실제 시작 calls | 성공 Teacher/Student | 출력 tokens | 종료·해석 |
|---|---:|---:|---:|---|
| 배포 전 GPU 충돌 | 0 | 0/0 | 0 | legacy 자동시작 workload 점유, 중복실행 보류 |
| v1 import | 0 | 0/0 | 0 | strict sandbox 임시경로 쓰기 불가, model load 전 실패 |
| v2 | 102 | 93/8 | 1,591 | nonempty abstention parser 실패; 부분점수 없음 |
| v3 clarification | 68 | 59/8 | 1,028 | 동일 계약위반, 첫 실패 중지 |
| v4 grammar | 210 | 124/85 | exact null / 관측하한3,216 / charge3,460 | 3,600초 상한 종료; paired coverage 불완전 |
| v5 optimized grammar | 248 | 124/124 | 3,802 | 완전채점, Teacher gate 음성 |
| order answer-first | 248 | 124/124 | 3,802 | 완전채점, 동일 deterministic control |
| order abstain-first | 248 | 124/124 | 3,058 | 완전채점, 유용성-거절 tradeoff 음성 |

실제 생성6회의 시작 calls합계 **1,124**. 앞2개 실패는 생성 실행수에 더하지 않는다. 실패한 active call과 strict parser 때문에 시작 calls와 성공합계가 같지 않다. 비용은 null이며 토큰·생성경로 elapsed가 Azure 청구액이나 순수 kernel 시간은 아니다. `trials/`는 각 원본 status·설정·strict stop reason·완전채점 결과를 연결한다. status의 `pending_local_scoring`은 생성 종료 당시의 원형으로 보존하고, 후속 score 결과를 별도 필드로 연결했다.

## 4. FT / A / B / C의 실제 경계

| 경로 | 실측·선택·보존 | 완료성 |
|---|---|---|
| Teacher 과제적응(FT) | 단일 연구실행 계보, seed42; 역사적600 CE0.09499394191815544 / best500 CE0.0934396254011231 | 적응 후 QA·Teacher gate 미완료. 새 복구 기준은 인증 g500; local600을 recoverable Blob600으로 세지 않음 |
| A 독립 gold-response CE | 1,000update, train charge806,091tokens / validation11회 charge1,094,731tokens; step0 CE0.16465599801956435 → best900 CE0.08195323988509474; latest1000 CE0.0897350995161432 | latest1000/best900 게시 인증. 선택best 복원평가1회·99,521 budget tokens 잔여, arm complete=false |
| B Teacher-response CE | 실행0 | Teacher 우위·응답 provenance/품질검수 선행, 미실행 |
| C B와 같은 응답/prefix+KL | 실행0 | B↔C가 KL 추가효과. 비교 자체 미완료 |

A step0→best의 dev CE 감소 **50.22760125910681%**, step0→latest 감소 **45.5014693691991%**. latest1000은 best900보다 **9.495487477931652% 악화**했다. 이는 저장 개발손실의 실제 진척/후반 악화이지 생성 QA 정답률 향상·KD 성공·독립 heldout 일반화가 아니다. token budget charge는 supervised answer-token 수와 같다고 가정하지 않는다.

실제 동결 policy는 양쪽 cap1000/lr0.0001/accumulation1/clip1/seed42를 보존한다. Teacher policy deterministic=false, Student A=true라는 차이도 원형으로 남겼다. A/B/C의 동일통제 설계가 구상됐다는 사실이 실제 paired 비교 완료를 뜻하지 않는다. Teacher FT를 자기생성 응답학습이나 Student KD로 재명명하지 않는다.

근거: `teacher_last_observed`의 `/runtime~1continuation~1native~1budget.json/projection`, `student_initial`의 `/history/0/response_ce`, `student_last`의 `/nested`, `student_published`의 `/best`·`/latest`, `not_arm_completion`의 `/complete`·`/pending`, `pause_status`의 `/student_A`·`/student_B_C`. 공개 `training/`에 A cadence11점 및 Teacher 저장관측100..600 6점을 보존했다. Teacher step0 수치는 이번 범위에서 미취득이며 임의 보충하지 않았다.

## 5. 자료·노출·일반화 한계

- KLUE MRC **공식 train** revision `349481ec73fff722f88e0453ca05c77a447d967c`의17,554행 전체를 기계검증·전역군집화 후 내부1,000행 채택. train794/dev124/sealed82, 군집603/90/68. 공식 validation/test 성적표가 아니다.
- source별 hankyung619/acrofan165/wikipedia216. 잘못된 행1,332 및 invalid 포함군집 관련2,839행 제외에 따른 선택편향을 남긴다. source+title document proxy와 5char Jaccard0.85는 의미중복 완전 제거가 아니다. 기록된 교차분할 군집/문서/정규문맥 overlap0을 전체 누수부재로 과장하지 않는다.
- 공식 train 전체 노출은 later sealed82의 최초 구축 노출도 포함한다. 보호 payload를 새로 읽거나 hash하지 않았다. 과거 flags=false나 결과파일 부재로 역사적 scoring/nonaccess를 증명하지 않는다. 공개 train의 사전학습 오염은 unknown이고 dev 반복사용은 선택편향을 갖는다.
- impossible의 upstream nonempty span은 plausible distractor일 수 있다. 이를 정답이나 Teacher target으로 자동 채택하지 않는다. 사람의 전체 의미/의학 검수는 별개다.
- CC-BY-SA 카드만으로 뉴스 원천의 ML연구 목적 조건, 원문재배포/서비스 배포권을 없애지 않는다. 원문과 bound membership은 패키지에서 제외했다.
- taskprep는 NSMC128행 preparation read→111개 smoke, train89/dev6/sealed16 및 노출된 비의료 authored dev3의 **CPU 합성 plumbing**이다. 회귀20개 통과·A/B/C replay는 실제 pretrained모델·Teacher응답·QA KD 효과가 아니며 durable모델 checkpoint0이다.

근거: `cohort` source `/counts`, `/split_clusters`, `/cluster_stats`, `/exposure`, `/limitations`; `prep_scope`, `prep_closure`, 별도 handoff5–22행. 이번 감사에서는 원문 데이터나 fixture 내용도 열지 않았다.

## 6. 실패 이유와 어려웠던 환경 — 사실과 가설

**확인된 실패:** GPU legacy autostart 충돌, strict sandbox tempdir 부재, v2/v3 거절 JSON 계약위반, v4 시간상한, Spot allocation의 `OverconstrainedAllocationRequest`, 게시/복원 경계. 실제 metadata compact23,677B가 기존 reader16,384B를 초과한 경계도 별도 기록했다. 이런 인프라/인증/용량 경계는 모델 학습효과의 음성검정이 아니다.

Teacher 복구에서는 g500 복원 뒤 새 update0/charged1/pending을 유지했다. 보존된 read-only 진단은 CPU 한 thread busy·IO/메모리 안정·관측 GPU utilization0와 optional FLA 빠른경로 부재/reference fallback을 지지한다. **정확한 장기 실행 frame은 미확정**이다. 첫 CUDA solver/JIT 초기화나 host loop는 가설이며 “FLA가27분 stall 전체의 원인/해결책”이라고 확정하지 않는다. 후보 FLA 설치·CUDA6 case는0회였으므로 수정 성공으로 보고하지 않는다. 후속 Agent/접근 장애의 정확한 guest 원인도 미확정이다.

이 taskQA의 보존 근거에 담긴 11월 8일 연구 중지는 연구보존·서비스 집중을 위한 사용자 지시였다. 이후 전체 연구/서비스 자동 실행도 사용자 요청으로 중지했다. 이번 정리에서 추가 실험은 하지 않았으며 미완료를 KD 실패나 무능력으로 바꾸지 않았다. 의료안전·실어르신 효용·양자화 전체앱 RAM/지연/발열·임상/상용 서비스 준비는 미검증이다.
