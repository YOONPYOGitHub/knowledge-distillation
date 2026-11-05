> Public portable edition. The Korean scientific report below is preserved verbatim from the completed report. Local publication scope, portable file mappings and historical-only evidence limitations are in [EVIDENCE.md](EVIDENCE.md). Results and paired effects are in [the results bundle](../../../results/published/generic-kd-v1/README.md); bare filenames in the historical body are archival references, not promises that every operational file is shipped.

# 한국어 기본 KD 3계열 통합보고

작성: 2026-10-31 KST. 기본 generic-CPT 실험 종료 및 통합보고 출판본. 수치·과학 주장 독립검토와 추가 metadata 주장 검토를 반영했다. 검토 범위와 신규 payload 인증이 아닌 한계는 EVIDENCE에 명시한다. task-aligned 연구 및 실버케어 서비스 완료 보고가 아니다.

## 1. 연구 질문과 판정

동일 Student에서 정답 토큰만 학습하는 CE 대비 Teacher 분포를 추가하는 KD가 한국어 평문 예측을 개선하는지 조사했다. CE는 교차엔트로피, KD는 지식증류, PPL은 낮을수록 좋은 토큰 예측 난이도 지표다. Qwen→Gemma→Liquid 순서로 단일 A100 Spot GPU에서 수행했다.

최종 test의 CE→KD 상대 PPL 변화(음수=개선)는 아래와 같다. 이 값은 서로 다른 tokenizer 사이의 품질 순위가 아니다.

- Qwen, seed42/43/44: −0.260352% / −0.539949% / −0.501996%. 고정 Teacher 조건에서 세 Student seed 모두 소폭 개선.
- Gemma: +0.847238% / +0.607555% / −0.278793%. 두 seed 악화, 한 seed 개선으로 일관된 KD 우위 없음.
- Liquid: −0.096916% / −0.296836% / +0.076152%. 두 seed 개선, 한 seed 악화로 일관된 KD 우위 없음.

세 계열을 통틀어 KD의 일반적 우월성·실용적 차이·통계적 유의성을 확증하지 않는다. 모든 수치는 `paired_effects.csv` 및 `results.csv`의 원본 집계에 근거한다.

## 2. 모델과 세대

저장소의 모델 설정은 smoke 기본값이며 정식 실험의 실제 예산·학습 seed는 frozen plan/report로 확인해야 한다. 다음은 고정 모델 revision이다.

- Qwen Teacher `Qwen/Qwen3.5-9B` @ `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; Student `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Gemma Teacher `google/gemma-4-12B-it` @ `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`; Student `google/gemma-4-E2B-it` @ `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- Liquid Teacher `LiquidAI/LFM2.5-24B-A2B` @ `3fcfdbfa1a1b6d8166846a6323e69f1e1254440d`; Student `LiquidAI/LFM2.5-2.6B` @ `654f9463ce32b05d0429d76fe1f580b27d4c1ac0`.

Liquid Teacher의 A2B 표기는 활성 규모이지 전체 24B 가중치가 2B라는 뜻이 아니다. 실제 최신 공개 모델·라이선스 조사 근거는 기존 감사문서에 있으며 이번 보고작성 tick에서 다시 조회하지 않았다. 배포 권리 및 gating은 재배포 시 별도 확인 대상이다.

## 3. 공통 데이터와 범위

FineWeb2 한국어 원문에서 공통 코퍼스 v1을 한 번 구성하고 세 계열에 같은 문서 ID·정제·분할을 적용했다. canonical manifest SHA-256은 `53e4e887ec898c2cce985a8027f0dec33360761ba16f705a3c95d244887f904a`다. train 6,814문서 및 validation 851문서 기록이 있다. exact dedup과 이메일·전화번호 휴리스틱 정제이며 완전한 개인정보 제거·근접중복 제거라고 주장하지 않는다. 원문 manifest 및 각 frozen prepared identity의 전체 연결은 최종 EVIDENCE에서 완결해야 한다.

공통 원문이더라도 tokenizer별 패킹·경계 손실·tail 및 실제 scored tokens는 다르다. 최종 test 유효 토큰은 Qwen 669,921, Gemma 739,417, Liquid 733,285다. 공통 raw ID만으로 동일 predicted-byte coverage가 입증되지는 않으므로 raw PPL 또는 근사 BPB로 계열을 순위화하지 않는다. 과거 200문서 Qwen 파일럿과 소형 GPU 진단은 공통 코퍼스 결과와 분리한다.

## 4. 방법·선택·seed

LoRA 기본 설정은 rank8/alpha16/dropout0.05, seq512, batch1이다. BF16 backbone/logits와 PEFT가 유지하는 native FP32 vanilla LoRA A/B 저장을 구분하며 저장 adapter를 강제 캐스팅하지 않았다. 평가 CE는 shifted FP32 합계/유효 토큰으로 계산했다. 상세 deterministic/TF32/CUBLAS 정책은 각 frozen source·plan을 기준으로 하며 이번 초안은 그 모든 필드를 재전사한 최종 manifest가 아니다.

Teacher를 한국어 train에서 먼저 적응하고 validation best를 동결했다. Student 탐색은 validation만으로 설정을 고르고 정식 공통 탐색은 계열당 두 round에서 닫았다. 각 round CE4/KD4 후보, arm당 최대3,000 update, validation200/checkpoint100, patience3/min_delta0.0001의 frozen plan 기록을 사용했다. 상한완료는 수렴의 증명이 아니다. CE에도 독립 LR 탐색을 주었으며 동일 update가 동일 GPU 비용을 뜻하지 않는다.

최종 CE/KD 설정을 고정한 뒤 Student seed42/43/44를 비교했다. seed42는 선택에 사용한 기존 실행을 재사용했으므로 새 독립 실행으로 더 세지 않는다. 각 seed 내 초기 adapter는 CE/KD 동일, 서로 다른 Student seed와 raw split seed는 구별한다. Teacher는 계열별 하나로 고정되었으므로 세 Teacher 반복 결과가 아니다. 선택 최종 checkpoint·실제 LR/온도/CE가중치/방향은 각 원본 selection/contract로 연결해야 하며 관련 정합검증 이력은 RUNBOOK 및 계열 closure에 보존되어 있다.

### 4.1 독립검토 후 보완한 실제 설정·종료 해석

보존된 seed43/44 report projection에서 확인한 선택쌍의 CE/KD 학습률은 Qwen `8e-5`, Gemma `1.5e-4`, Liquid `1.25e-4`다. KD의 alpha는 각각 `0.7/0.7/0.9`이며 **CE 가중치**를 뜻한다(KD 가중치는 1-alpha). 세 계열 모두 `forward_kl`, 온도 `1.0`이다. 근거는 `training-reports-compact-20261031T0923.json`의 해당 report/trial 필드와 seed42 6건의 `seed42-selection-verification-20261031T1206.json`이다. `selected-checkpoint-links.csv` 21행은 Teacher3+Student18 선택의 모델/revision·계약·cohort·checkpoint manifest/COMPLETE 및 계열별 선언된 수치 source7개를 연결한다. 이는 metadata 연결 검증이며 전체 checkpoint payload·Blob·전체 배포 capsule을 새로 읽어 검증한 것은 아니다.

실제 선택 checkpoint optimizer는 `torch.optim.adamw.AdamW`, betas `(0.9,0.999)`, eps `1e-8`, weight_decay `0.01`이며 Teacher LR은 `2e-5`다(`selected-contract-scalars-20261031T1112.json`, `selected-details-20261031T1129.json`). Student 계약의 수치정책은 deterministic_algorithms=true, cuDNN deterministic=true/benchmark=false, TF32=false, CUBLAS workspace `:4096:8`이다. Teacher identity 위치의 numerical_policy=null은 그 위치에 키가 없다는 뜻이며 정책 부재 증거가 아니다. 저장 projection에는 scheduler 증거가 없으므로 scheduler 사용 여부는 이번 연결 자료만으로 단정하지 않는다.

3,000 update는 공통 최대 예산이지 모든 후보의 실제 종료 step이 아니다. 예를 들어 Gemma Round2의 ce-4는 latest/best `2000/1400`, kd-4는 `2600/2000`, kd-2는 `3000/2800`이다(`runs.csv`). 완료된 조기종료 후보와 실행실패를 혼동하지 않으며, 최종 선택쌍의 결과와 전체 탐색 종료 분포를 구분한다.

## 5. Teacher 결과 — Student와 분리

정식 Teacher validation의 step0→best3000 PPL:

- Qwen: 15.102513→12.186575, 동일 validation 677,586 유효 토큰.
- Gemma: 7,388.332697→11.297211, 동일 validation 747,082 토큰.
- Liquid: 39.624945→18.655907, 동일 validation 740,950 토큰.

최종 test Teacher PPL은 각각 `results.csv`의 Teacher 행을 따른다. Liquid에서는 Teacher PPL `19.003929`가 최종 CE Student `17.544124–17.629134` 및 KD Student `17.557484–17.576804`보다 높았다. 즉 같은 계열·test 평문 예측에서는 적응 Teacher가 모든 최종 Student보다 열세였다. 이 사실은 KD 해석의 맥락이지만 혼재 결과의 원인으로 확정한 것은 아니다. Teacher 적응과 Student KD의 효과는 다른 질문이다. 특히 Gemma의 매우 높은 평문 초기 PPL은 기존 loader/공식 forward 진단에서 구체적 결함으로 확정되지 않았으며, 큰 적응폭을 대화능력 향상이나 KD 효과로 해석하지 않는다. 계열 간 Teacher 순위 역시 내리지 않는다.

## 6. Student CE 적응과 KD 증분

사전학습 Student→CE의 test PPL 감소율은 seed별 Qwen 약24.20–24.38%, Gemma 약99.7443–99.7451%, Liquid 약67.76–67.91%다. 이 큰 ordinary adaptation 효과와 CE→KD의 작은 추가 차이를 혼동하지 않는다. 모든 seed의 정확한 PPL 및 상대변화는 CSV에 남겼다.

최종 값은 validation-selected best를 고정한 뒤 측정한 test 값이다. 탐색 중 matched-step 곡선의 일시적 교차는 최종 우위로 세지 않는다. 과거 pilot/중간 checkpoint/진단값으로 test 최종결론을 대체하지 않는다. 결과가 혼재한 계열도 유한 예산 완료라는 의미에서 닫았고 test 이후 재튜닝하지 않는다.

## 7. 실행 수와 장애의 해석

Student 원본 report 및 실제 frozen plan seed가 대조된 행은 공통 코퍼스 60건(계열별 CE10/KD10), 과거 Qwen 파일럿16건(CE8/KD8)이다. 공통 계열당16건의 seed42 탐색과 seed43/44의 고정쌍4건이며 20개 독립 training seed가 아니다. 최종 선택3쌍은 탐색실행과 겹친다. 정식 Teacher3건과 역사적 소형 진단10건의 positive final_step report는 별도 부분표에 있다. 이 수는 모든 역사적 실행의 완전한 census나 전체 checkpoint payload 재검증 수가 아니다.

재개는 원래 run/trial 한 건으로 센다. Spot중단·권한/패킹 실패·누락 source·loader/dtype 평가기 실패·합성 forward/backward·CPU 테스트·복구 replay를 새 모델효능 실행으로 세지 않는다. historical JSON inventory 908개 중 파싱 불가3개는 빈 전송 artifact이며 실패 사유를 보존했다. JSON 개수는 학습 실행 수가 아니다. 학습 전 실패4건과 학습 후 evaluator dtype 장애1건은 `FAILURES.md`와 원문 bytes/SHA·오류문자열 검증 파일로 연결했다. 이는 전체 역사 실패 census가 아니며 실패를 숨기거나 정식 학습 실행으로 더하지 않는다.

## 8. heldout·보존·재현성

선택 후 최종 test가 열렸으므로 각 계열의 재튜닝은 금지한다. 무결성 SHA 읽기와 tokenizer packing·모델 scoring·사람의 내용 확인은 구별한다. 과거 모든 비접근을 artifact 부재만으로 입증할 수 없으며 관련 scoped exposure adjudication에 제한된 근거와 불확실성을 보존했다.

세 계열 최종8모델씩의 원본24 JSON을 회수하여 byte/SHA를 확인했고, FP32 CE합/토큰→PPL 재계산과 기존 metrics 대조를 수행한 기록은 `aggregation-verification.json`이다. 각 계열 closure에는 결과/source/bundle/완료로그의 관리ID 인증 Blob readback, payload·manifest·COMPLETE byte/SHA 및 COMPLETE.manifest_sha256 검증 근거가 있다. 이번 문서 작성에서 모델을 다시 학습하거나 test를 다시 채점하지 않았다.

문서/cluster별 per-unit losses로 bootstrap을 재구성하지 않았으며 aggregate만으로 문서 신뢰구간을 만들지 않는다. Student3seed의 작은 차이가 실용적/의료적으로 의미 있다는 수치기준도 사전 동결되지 않았다.

## 9. 비용·온디바이스·실버케어 한계

실제 학습 GPU초·중단포함 wall time·Azure 청구액의 완전한 근거가 없어 미측정/확인불가로 닫는다. service 관측창·scheduler 간격을 실제 훈련시간으로 만들어 넣지 않는다. KD는 Teacher inference 비용이 추가되므로 equal-update에서 equal-cost 효율 결론을 낼 수 없다.

양자화 export, 실제 6–8GB RAM 기기의 전체 앱 메모리·지연·발열·전력, 한국어 grounded QA·instruction-following·안전·사람에게 의뢰·개인정보·접근성은 미검증이다. A100 학습 메모리 probe와 PPL 개선은 이러한 검증을 대신하지 않는다. 기본 generic-CPT 종료가 의료안전 승인이나 실버케어 서비스 readiness를 뜻하지 않는다.

## 10. 후속 연구와 완료 gate

문헌 기반 제안은 별도 research-support 큐에 있으며 기존 실험을 소급 변경하지 않는다. 다음 task-aligned 설계는 A 정답 CE → B Teacher응답 CE → C 동일응답+KL을 우선 비교하고, 검증된 경우에만 D on-policy를 별도 제어한다. 고정 corpus prefix의 reverse token KL은 MiniLLM/GKD sequence-level on-policy 실험과 같지 않다. 새로운 cohort/목적함수/Student는 새 CE/KD 대조군과 유한예산이 필요하다.

별도 taskprep의 CPU 기능시험·NSMC smoke는 grounded QA/안전 gold 또는 실제 Teacher 생성/GPU 과제형 학습 결과가 아니다. 기본 통합보고의 선행조건으로 추가하지 않는다.

**출판 검증:** 수치·과학 주장 검토 `deleg_070ee302` 및 추가 주장 검토 `deleg_102b9bc5`를 반영했다. 선택 metadata21건·seed42 6건 및 한정된 실패 분류는 연결했으나 전체 payload 신규 검증으로 승격하지 않는다. 최종 파일별 bytes/SHA는 `manifest.json`, 전달 묶음의 검증은 별도 publication receipt에 기록한다. 남은 task-aligned 연구와 실기기·의료안전 검증은 본 기본실험 종료와 구별한다.
