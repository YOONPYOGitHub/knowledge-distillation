# 실버케어 보조 대화 연구: 실제 완료 결과와 미완료 범위

기록 정리일: 2026-11-14. 실험을 재시작하거나 새 답변을 생성하지 않고 보존된 결과를 정리했다. 아래 수치는 **비의료 보조 대화의 내부 개발평가**이며 의료 QA·안전성·독립 test 결과가 아니다. 기본 한국어 평문 연구 및 KLUE 기반 taskQA와 별도 cohort다.

## 1. 무엇을 실제로 했는가

- 보조 대화 자료 52행 중 train 41행 / 내부 dev 11행을 사용했다. 임상 수용행은 0이다. 원 참여자·증강 독립성이 입증된 대규모 분할이 아니다.
- Teacher는 `Qwen/Qwen3.5-9B` (`c202236235762e1c871ad0ccb60c8ee5ba337b9a`), Student는 `Qwen/Qwen3.5-4B` (`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`)다.
- LoRA rank8 / alpha16 / dropout0.05, thinking=false로 비의료 말투 적응을 진행했다. 학습이 실제 실행되었다는 사실과 의료 효능은 다르다.
- 41행 적응과 별도로 **같은 12행·동일 Student 초기 adapter·순서·seed42·각 12 update**에서 A=제공 정답 CE, B=Teacher 응답 CE, C=동일 Teacher 응답+KL을 비교했다. 이 12개는 뒤에서 설명하는 train41 Teacher 응답 검토 후 제한 수용된 subset이다. 41-update CE를 12-update KD의 공정한 대조군으로 쓰지 않는다.

원본 metadata의 정확한 공개 projection과 SHA-256은 [service-results.json](evidence/service-results.json)의 `teacher41`, `student41`, `matched12-ce`, `matched12-kd`에 있다. 원문 대화와 checkpoint는 포함하지 않는다.

## 2. 41행 적응 — 손실은 개선했지만 품질 승인은 아님

| 모델/arm | 초기 dev CE | 선택 dev CE | 초기 dev PPL | 선택 dev PPL | 선택/최종 step | dev 유효 토큰 |
|---|---:|---:|---:|---:|---:|---:|
| Teacher 9B 보조 대화 CE | 2.757958938826376 | 1.694984891521397 | 15.76762739041548 | 5.446563675300413 | 41/41 | 201 |
| Student 4B 정답 CE | 2.849849890713668 | 1.7288385552553396 | 17.28518697873459 | 5.634106403608484 | 41/41 | 201 |

두 실행 모두 max_steps 종료다. 0/10/20/30/40/41 step의 전체 저장 dev 곡선을 evidence에 보존했다. 기록된 학습 구간 elapsed_seconds는 Teacher `62.67426171596162`, Student `61.929591812193394`이고 peak_vram_bytes는 각각 `18712146432`, `9238711808`이다. 이는 그 실행에서 계측한 값으로 전체 준비·전송·중단을 포함한 소요시간이나 모바일 앱 메모리가 아니다.

**왜 이 값을 서비스 성공이라고 하지 않는가?** Teacher의 실제 내부 dev 11건을 본 당시 LLM 보조 검토는 기본 수용 3건 / 수정 필요 8건이었다. 감정 불일치, 지지보다 사건 재진술에 치우침, 근거 없는 동기·인과 부여, 반복, 관계·불확실성 처리 미흡이 기록됐다. 이는 임상 전문가 검토나 객관적 정확도 점수가 아니다. dev 손실 감소만으로 Teacher의 목표 응답 품질 및 Student에 대한 우위가 확인되지 않았다. Teacher 응답을 쓴 임상 학습 승인으로 확대하지 않는다. 근거: `service-state-snapshot`의 `/auxiliary_teacher_dev_review/*`.

### 2.1 후속 생성 정책 전후 검토 — 전체 품질 향상 미입증

초기 dev11 검토가 마지막 검토는 아니다. 같은 선택 adapter에서 생성 정책을 바꾼 뒤 11개 출력 모두 달라졌고, 11/11 EOS 종료·cap0이었다. 추가 학습은 0 update다. 이어 `gpt-6.1-sol`로 11쌍을 AB/BA 순서로 검토했다. **실제 HTTP judgments24건=11쌍×2순서+neutral calibration2건**이며 독립 문항24개나 평가자24명이 아니다.

| 당시 manifest의 A/B | 원래 정책 A | 변경 정책 B |
|---|---:|---:|
| 여섯 차원 전체 PASS | 1/11 | 1/11 |
| 전체 차원 label PASS | 49 | 47 |
| REVISE | 17 | 18 |
| UNCERTAIN | 0 | 1 |

문항 단위 선호는 A4 / B2 / 동률2 / 불확실3, 순서 일관 사례는 8/11이었다. 비기계적 지지의 PASS는 A1/B3처럼 일부 차원에서 달랐지만 **변경 정책의 전체 품질 향상은 입증되지 않았다**. 최초 기본 수용3/11과 엄격한 전 차원 PASS1/11은 기준·절차가 다르므로 악화율로 계산하지 않는다. 임상·사람 승인 및 이 dev응답의 학습 수용은 false다.

### 2.2 train41 응답 검토43건과 12개 제한 수용

별도로 고정 Teacher의 **train41 응답+neutral2건=총43호출**에 대한 실제 LLM 검토가 완료됐다. HTTP200/VALIDATED는 43/43, 기록된 elapsed_seconds는 `251.977`이다. IN_SCOPE이고 원래 여섯 차원이 모두 PASS인 train응답 **12개 수용 / 29개 제외**가 뒤의 owner admission에 기록됐다.

차원별 PASS/REVISE는 감정 충실성29/12, 비기계적 지지14/27, 쉬운 언어40/1, 관계 보존39/2, 불확실성 보존38/3, 근거 없는 원인 회피38/3이다. 차원별 실패는 겹치므로 합해서 제외 건수로 쓰지 않는다. 수용12개는 뒤의 matched12 cohort와 연결된다.

**시간/범위 구분:** 원래 dev11 불수용 → 검토 ledger 완료(아직 수용 전) → owner의 비의료 train응답12개 per-case 제한 수용 → matched12 A/B/C 실제 실행 순서다. 과거 `responseCE_KD_started=false`를 후속 실행까지 없었다는 뜻으로 쓰지 않는다. 12개 수용은 Teacher 전체 우위·train41 전체 수용·임상·사람 승인과 다르다. Teacher 우위는 여전히 false다.

이 추가 집계·원본 pointer/SHA는 [service-supplement.json](evidence/service-supplement.json)의 `policy-dev11-*`, `train41-review43-*`, `train41-acceptance12`, `later-quality-state`에 있다. 모든 원문 응답과 개별 검토 서술은 제외했다.

## 3. 12행 matched A/B/C — KL 효과와 정답 CE 비교를 분리

공통 초기 dev CE `2.849849890713668`, PPL `17.28518697873459`. dev 11행 / 유효 토큰 201, batch1, seq128, 한 epoch·각 12 update, Student AdamW LR `2e-5`. C는 forward KL / T=1.0 / alpha=0.5다. 원 metadata의 `vocab_size=0`을 그대로 보존하며 이를 어휘가 0개라는 뜻으로 해석하지 않는다.

| arm | 최종/선택 dev CE | 최종/선택 dev PPL | 실제 update |
|---|---:|---:|---:|
| A: gold CE | 2.571779222630743 | 13.089092144543514 | 12 |
| B: Teacher 응답 CE | 2.598018987854915 | 13.43709260760771 | 12 |
| C: 동일 응답 CE+KL | 2.5730011214071244 | 13.105095465446862 | 12 |

상대변화는 `(나중 PPL / 기준 PPL − 1) × 100`이다.

- **B→C: −2.470751%**. 이 작은 내부 dev 실험에서는 KL을 추가한 C가 같은 Teacher 응답 CE인 B보다 낮은 PPL이었다.
- **A→C: +0.122265%**. C는 정답 CE인 A보다 조금 나빴다. 따라서 “KD가 일반 CE보다 우수하다”는 결론은 나오지 않는다.
- A→B는 +2.658706%다. Teacher 응답을 target으로 바꾼 영향과 KL 추가 효과를 섞지 않는다.

한 seed·12개 보조 대화·12 update·선택에도 쓰는 dev이므로 통계적·일반적 효과를 확증하지 않는다. Teacher 우위는 미검증이다. 학습 완료와 checkpoint 복원 기록은 수치 실행 근거이나 독립적인 임상 품질 승인과는 별개다. 해당 비교는 대규모 의료 증류 실험의 대체물이 아니다.

## 4. 실제 생성 평가가 남긴 부정적/제한 결과

### 4.1 별도 Base↔CE41 dev11 종료 양상

추가 과제학습 전 Base 4B는 dev11에서 11/11 cap·EOS0, CE41 Student는 11/11 EOS·cap0이었다. 총 출력은 각각 704/179토큰, generation_seconds 합은 `111.51536888582632` / `27.952281524892896`이었다. 함께 기록된 Teacher11개는 앞선 결과의 재사용이지 추가 독립 생성이 아니다. **길이/종료 양상 변화**는 확인되지만 의미 품질 향상·동일 길이의 인과적 속도 우위는 미검증이다. 다음 matched12의 dev33과 다른 checkpoint/실행이다. 근거: supplement의 `base-ce41-dev11-generation-comparison`.

### 4.2 matched12 dev33 — 생성 완료, 새 출력 내용 검토 미완료

12행의 세 arm에서 dev 11건씩 **33개 답변 생성은 완료**됐다. 이후 metadata 분석에 따르면 모든 출력은 64토큰을 사용했고, A 10/11, B 11/11, C 11/11이 EOS 없이 상한에 도달했다. A의 1건은 경계에서 EOS로 끝났다.

- EOS 없는 상한 종료는 **종료 metadata**다. 원문을 다시 읽지 않은 이번 정리에서 이를 곧바로 의미적 오답/불완전 판정으로 바꾸지 않는다.
- 세 모델이 같은 문항에서 모두 EOS 종료한 공통 사례는 **0건**이다.
- 새 33개 답변의 의미·정확도·안전성 검토는 완료되지 않았다. 완전한 답변의 품질 순위를 만들 수 없다.
- arm별 한 번의 순차 실행에서 관측된 token rate/문항별 시간차를 인과적 속도 우위, TTFT 또는 모바일 전체 앱 지연으로 해석하지 않는다.

원시 QA를 제외한 coverage·상한/EOS·paired timing metadata는 `dev33-metadata`에 보존했다. 평가를 끝내지 못한 사실은 성능 향상 0%를 측정했다는 뜻이 아니다.

## 5. 앱 기능시험에서 성공한 범위와 실패한 범위

과거 CE41의 **비공개·비의료 단일 요청 UI→API→실제 모델 생성→화면 표시** 기능시험은 성공 기록이 있다. 제한된 입력에서 20 input tokens / 13 output tokens / `2.3031760221347213`초 / EOS 종료가 기록됐다. 이는 1회 native 생성의 계측이며 지속 서비스 SLA나 일반 속도 측정이 아니다. 실제 browser click과 native 출력 hash 일치 기록은 있지만, 임상 release·public release·quality acceptance는 모두 false다.

당시 경로는 비공개 답변 JSON을 저장했다. 따라서 “비공개 E2E 성공”을 입력/출력 무저장·의료안전·현재 서비스 전환 요건 충족으로 확대하지 않는다. 후속의 상시 실행·후보 전환/rollback 경로는 별도 설계이며 실제 switch 성공은 확인되지 않았다.

후속 후보의 blocker는 active systemd manager/cgroup·전체 descendant 종료·cache publisher custody·동시 RAM/VRAM peak·실제 tokenizer/model constructor identity 검증 등이다. 이들은 해당 애플리케이션의 구현/안전성 요건 또는 미검증 사항이다. **제공자가 모델 실행을 금지했다는 증거가 아니다.** 50GiB 디스크 예산을 RAM 제한으로 잘못 해석하지 않는다. 당시 이미 동작한 단일 요청 경로와 새 상시 전환 경로를 구별한다.

### 5.1 표준 PEFT 3-arm 실제 복원·추론과 Azure 후보 단일 추론

- **표준 PEFT 실제 실행:** 후속 parent receipt는 세 arm 모두 실제 모델 로드·표준 PEFT import·neutral 생성1회씩 완료했다고 기록한다(`fixture_only=false`). arm별496 / 총1,488 tensors를 확인했고 추론 후 비교도 일치했다. 입력18/출력4토큰, generation_seconds는 A `0.9558313849847764`, B `1.05212672194466`, C `1.0485873010475188`이다. 새 학습·dev 재생성은0, parent acceptance=true지만 품질·임상·service promotion=false다. 형식 변환/CPU fixture 준비에만 그친 것으로 쓰면 부정확하다. 한 neutral 요청으로 속도 순위를 만들지 않는다.
- **Azure CE41 model-only 후보:** 원본 receipt는 표준 PEFT496 entries 복원, 출력10토큰·EOS=true·`1.4458181729996795`초 생성·peak VRAM `11590027264` bytes의 실제 단일 추론을 기록한다. 전체 worker elapsed는 `9.821702630000345`초다. `candidate_app_integrated=false`, `old_current_kept=true`, promotion=false, service pointer rollback=false다. 즉 **후보 load/단일 추론 성공이지 앱 전환·rollback 성공은 아니다.** 해당 경로의 raw input/output 미보존 기록을 앞선 다른 UI E2E 경로에 소급하지 않는다.

근거: supplement의 `matched12-standard-peft-native-parent`, `matched12-standard-peft-native-evidence`, `azure-ce41-native-model-neutral`, `main-service-gpu-execution-state`.

### 5.2 개인정보 이력의 한계

과거 보존 Student dev출력11행이 도구 stdout에 표시된 incident가 있었다. 기록은 raw_in_tool_stdout=true, new_model_api_transmissions=0, credential_exposure_observed=false다. 따라서 전체 이력의 **“원문 노출0·항상 무저장”을 주장하지 않는다**. 이번 출판에는 그 원문을 다시 열거나 복제하지 않고 집계·발생 사실만 남겼다. 새 API전송0과 과거 stdout노출0은 다르며, 이를 dev33 전체 노출로 확대하지 않는다. 근거: supplement의 `historical-raw-stdout-incident`.

## 6. 최종 판정

- **측정된 성공:** 제한된 비의료 보조 대화의 실제 CE 학습, dev 손실 감소, 12응답 제한 수용과 matched12 A/B/C 실행, dev33 생성, 표준 PEFT3 실제 복원/추론, Azure 후보 단일 추론, 과거 단일 요청 비공개 E2E.
- **긍정/부정이 함께 있는 결과:** B→C의 dev PPL은 개선했지만 C는 A를 이기지 못했다. Teacher의 낮아진 PPL과 정책 변경도 전체 응답 품질/우위 승인을 뜻하지 않았다. 별도로12응답의 좁은 비임상 학습 수용은 실제 이루어졌다.
- **미완료:** 임상 QA 수용·대규모 의료 학습·독립 test·새 matched12 dev33의 의미/안전성 비교·상시 서비스 전환/rollback·양자화·실제 저사양 기기 검증.
- **종료 방향:** 연구/서비스 자동 실행과 정기보고는 사용자 요청으로 중지했다. 이 문서는 새 실행이나 재개 승인이 아니다.
