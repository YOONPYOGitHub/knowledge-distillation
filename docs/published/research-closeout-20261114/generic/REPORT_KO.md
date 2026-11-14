# 한국어 generic KD 출판 종료 감사

## 판정
**단일 smoke의 KD 패배로 종료 결론을 내리면 잘못이다.** 최종 공통 cohort 3계열×Student seed42/43/44에서는 Qwen만 세 쌍 모두 소폭 개선했고, Gemma·Liquid는 개선/악화가 혼재했다. 일반적 KD 우월성·통계적 유의성·실용 효과·한국어 QA/의료안전/실기기 우위를 입증하지 않는다.

원본 저장소 HEAD `dbdaa7d718f07fc8d68cbd4593a5c9eb2e9acb88`의 보존 메타데이터만 읽었다. 기존 dirty/untracked 파일을 보존했다. 학습·채점·모델/코퍼스/QA payload 읽기·네트워크·cloud 변경·commit/push 없음. 새 산출물은 지정 scratch에만 기록했다.

## 1. 최종 cohort·모델·선택
- `results/published/generic-kd-v1/cohort-manifest.json` → `/identity/{dataset,revision,subset,seed}`: `HuggingFaceFW/fineweb-2`, `af9c13333eb981300149d5ca60a8e9d659b276b9`, `kor_Hang`, split seed42. `/counts`: train6814/validation851/test851. `/cleaning`: exact dedup과 이메일/휴대전화 휴리스틱 제외이며 완전 PII/근접중복 제거 아님. 개별 제외 source IDs/text hashes는 새 extract에서 제외했다.
- `results/published/generic-kd-v1/selected-metadata.json`의 `/selections/i/{model_id,model_revision,settings,selected_step}`가 실제 선택 근거다. Qwen Teacher→Student: `Qwen/Qwen3.5-9B`→`Qwen/Qwen3.5-4B` (Teacher i=6; Student i=0–5). Gemma: `google/gemma-4-12B-it`→`google/gemma-4-E2B-it` (i=13; i=7–12). Liquid: `LiquidAI/LFM2.5-24B-A2B`→`LiquidAI/LFM2.5-2.6B` (i=20; i=14–19). 정확한 revisions는 evidence.json에 원값 그대로 있다. Gemma는 `-it`; 모든 모델을 일괄 Base 또는 Instruct라고 부르지 않는다. Liquid A2B는 활성 규모이며 전체 가중치는 24B 표기다.
- Student CE/KD LR은 Qwen `8e-05`, Gemma `0.00015`, Liquid `0.000125`; KD alpha=`0.7/0.7/0.9`는 **CE 가중치**, KL은 1-alpha. 모두 forward KL, T=`1.0`, 선택 step=`3000`. `/selections/i/optimizer_groups/0`: AdamW betas `[0.9,0.999]`, eps=`1e-08`, weight_decay=`0.01`. Teacher LR=`2e-05`.
- 같은 seed CE/KD 초기 adapter hash가 일치한다. seed42는 선택에 사용한 run 재사용, 43/44만 추가 고정설정 반복이다. 계열별 Teacher1개를 고정했으므로 Teacher 변동성 반복이 아니다. split seed와 training seed를 구별한다.

## 2. 최종 heldout: 원값과 증분
아래는 `results/published/generic-kd-v1/paired_effects.csv`의 `(family,seed)` 행 원값이다. Δ=`(KD/CE-1)*100`, 음수 개선. family 간 순위표가 아니다.

| family | seed | CE PPL | KD PPL | Δ % |
|---|---:|---:|---:|---:|
| qwen35 | 42 | 14.862177175245707 | 14.823483214906739 | -0.26035189785932333 |
| qwen35 | 43 | 14.883492218201454 | 14.80312891811785 | -0.539949219614777 |
| qwen35 | 44 | 14.898702584432938 | 14.823911764968404 | -0.5019955196815529 |
| gemma4 | 42 | 12.261580431171273 | 12.36546514457793 | 0.8472375481268557 |
| gemma4 | 43 | 12.278857508137035 | 12.353458287881676 | 0.60755473133558 |
| gemma4 | 44 | 12.296883945022318 | 12.262601114468014 | -0.27879282839113895 |
| lfm25 | 42 | 17.59161475670737 | 17.57456569036598 | -0.09691586916368289 |
| lfm25 | 43 | 17.629134005960093 | 17.57680442065329 | -0.2968358246588476 |
| lfm25 | 44 | 17.544124247691666 | 17.55748447157706 | 0.07615212761133971 |

정확한 JSON 근거 경로 규칙: `results/published/generic-kd-v1/final/{family}--{ce|kd}-seed{42|43|44}.json`의 `/statistics/ce_sum`, `/statistics/tokens`, `/status`, `/protocol_sha256`. CE=`ce_sum/tokens`, PPL=`exp(CE)`이며 원본 `results.csv`의 CE/PPL과 대조했다. 원래 `results/research/...` 경로·SHA는 evidence.json의 `/final_results`에 각 행별로 있다. 최종 24 JSON 모두 publication↔original 동일 SHA/내용과 산술을 확인했다.

| family | test 유효 tokens | pretrained Student PPL | 적응 Teacher PPL |
|---|---:|---:|---:|
| qwen35 | 669921 | 19.654976931798476 | 12.191398277069625 |
| gemma4 | 739417 | 4809.47717309216 | 11.235304410005034 |
| lfm25 | 733285 | 54.67775818637094 | 19.00392860091 |

이 표의 JSON은 같은 디렉터리 `{family}--pretrained-student.json`/`{family}--teacher.json`, 동일 pointers다. Student pretrained→CE의 큰 적응폭과 CE→KD의 작은 증분은 별개다. 특히 Liquid Teacher는 같은 계열 test에서 모든 최종 CE/KD Student보다 높은 PPL이다. 혼재 결과의 원인으로 단정하지는 않는다. Gemma 초기 평문 PPL의 큰 감소를 대화능력이나 KD 효과로 해석하지 않는다.

## 3. 과거 evidence: 긍정·부정 결과를 모두 분리 보존
- **Qwen3.5 smoke**: `results/research/qwen35-smoke-v1/test-results.json` → `/metrics/student_ce/ppl`=`15.164709883737956`, `/metrics/student_kd/ppl`=`15.419995773890824`, token=`11753`; `/packing/documents`=`16`. `config.json`의 `/data/max_documents`=`200`, `/training/{seed,max_steps}`=`42/20`. train-source hash split의 소규모 GPU smoke이며 최종 공통 cohort 결론을 대체하지 않는다.
- **그 pilot 탐색 validation**: `results/research/qwen35-tuning-round1/metrics.json` → `/methods/ce/ppl`=`17.667033827809654`, `/methods/kd/ppl`=`17.73947862773708`, `/ppl_kd_worse_percent`=`0.41005638316824466`, `/test_evaluated`=`false`. `qwen35-tuning-round2/boundary-compact.json` → `/methods/{ce,kd}/best_validation/ppl`=`17.66469166548576/17.681919388937064`, `/test_evaluated`=`false`. 모두 작은 pilot validation이지 최종 heldout 아님.
- **이전 한국어 Qwen2.5 GPU**: `results/logs/20260823_080109/summary.json` → `/config/{teacher_model,student_model}`=`Qwen/Qwen2.5-3B`→`Qwen/Qwen2.5-1.5B`; `/results/Student (FT)/ppl`=`11.800220931488933`, `/results/Student (KD)/ppl`=`14.225834324376834`로 악화. 7B→1.5B `qwen7b-korean-4way-v1/summary.json`에서는 `11.813621536575697/14.353540872934012`, T4 tokenmean v1/v2 KD=`18.78493479498928/15.710539669384737`로 역시 악화. 경로 접두어는 모두 `results/logs/`; 같은 결과 pointer 규칙이다.
- **이전 긍정**: `results/logs/qwen7b-korean-reverse-v3/summary.json`의 같은 FT/KD pointers=`11.813621536575697/11.6848729040593`. 단, 기존 FT Student 초기화·reverse KL·alpha/LR 등을 함께 바꿔 reverse KL 단독 인과효과가 아니다. 공유 FT/Teacher를 새 독립 seed로 세지 않는다.
- **H100 controlled 추가훈련**: `results/logs/h100-tuning-20260920/test_results.json` → `/test/student_ce_control_best/ppl`=`11.814911107554344`, `/test/student_kd_selected/ppl`=`11.353113405876357`, `/test/student_ce_at_kd_step/ppl`=`12.06861904233821`, tokens=`35955`. `/selection`은 validation-only freeze 후 test once라는 보존 선언이다. `summary.json` → `/manifest/training_seed`=`42`, `/manifest/gpu`=`NVIDIA H100 NVL`, `/manifest/base_config`은 Wikipedia `20231101.ko`, 1000문서, validation/test 각각0.05, 7B→1.5B, reverse KL/tokenmean. CE best step0와 기존 FT Student 출발, 단일 seed·작은 Wikipedia holdout·튜닝예산 대칭성 한계를 명시한다. FineWeb2 3계열과 합산 금지.
- **영어/MPS 역사**: evidence.json의 `/historical_log_summaries`에 14개 원본 summary 수치/모델/device와 pointers를 남겼다. GPT2→distilgpt2 WikiText는 한국어 실험이 아니다. `20260410_041631` FT/KD=`32.81794268364906/32.44255220049383`의 긍정도 있으나 앞선 음성 결과와 초기 PPL≈1의 masking/평가 의심 로그까지 보존했다. 원본 summary에 training seed 없는 경우 추정하지 않았다. CUDA 표기/보고 GPU는 실제 모델 실행의 보존 근거이며 CPU regression·합성 입력·replay를 언어 효능 반복으로 승격하지 않는다.

## 4. 프로토콜·유의성·출판 한계
- family별 동일 tokenizer/Student/heldout의 paired 비교만 지지한다. 공통 raw라도 packing 경계·tail·예측 byte coverage가 달라 cross-family token PPL 또는 근사 BPB 순위 금지. Teacher·Student 순위도 별개 질문이다.
- 계열당 2 search rounds, round당 CE4/KD4, arm당 cap3000, validation200/checkpoint100, patience3/min_delta0.0001. cap 완료는 수렴 증거가 아니고 완료된 early stopping은 실패가 아니다. validation 곡선 교차는 최종 test 승리로 세지 않는다.
- `/classification_counts` (`results/published/generic-kd-v1/run-census.json`) 공통 Student60/정식 Teacher3, 역사 pilot16/diagnostic10은 범위 한정 실행 집계다. 60 독립 seed 또는 전체 역사 census가 아니다. resume 중복계수 금지.
- aggregate에는 문서/cluster별 loss가 없으므로 문서 bootstrap CI를 재구성할 수 없다. 세 Student seed, 그중 선택 seed42, 고정 Teacher 조건에서 유의성·모집단 일반화·실용 임계값을 확증하지 않는다. token 수를 독립 표본 수로 쓰지 않는다.
- test 개방 후 재튜닝 금지. artifact 부재나 `heldout_read=false`만으로 과거 비접근 전체를 입증하지 않는다. SHA 확인·packing·model scoring·사람 열람은 다른 접근이다.
- Student numerical policy는 retained identity에서 확인하나 Teacher null 필드는 정책 부재 증거 아님. scheduler 증거 없음. LoRA rank8/alpha16/dropout0.05, seq512/batch1은 보고 수준 설정과 계약 수준 근거를 구별한다.
- 실패 부록 `docs/research-campaign/reports/basic-integrated-v1/FAILURES.md`는 준비/권한/source/cache 실패와 학습 후 evaluator BF16 오류를 구별한다. 새 패키지의 safe error tokens도 failed optimizer trial census가 아니다.
- GPU초/정확 wall-time/청구액 및 KD 추가 Teacher 비용은 완전 집계 불가. equal-update≠equal-cost. 양자화/전체앱 메모리·지연·발열·grounded QA·안전·의료/서비스 readiness는 미검증이다. 과거 Blob readback은 과거 주장이지 이번 fresh 인증 아님.

## 5. 실제 재분석용 확대 패키지
`PACKAGE.md`를 입구로 한다. 기존 `results/published`의 최종 aggregate 24개는 확대 패키지에서 중복 복사하지 않았다.
- `research-trials.json`: 원본 report projection에 남은 completed Student76건(공통60+pilot16), trial 설정·초기/최선 validation·best/final step·종료 이유와 frozen plan seed를 연결했다.
- `research-projections.json`: scientific scalar만 허용한 metadata projection. 372 JSON source, 398 source sections를 370 leaf-map으로 dedup; 13,921 원래 leaf occurrence를 원본 pointer/decode-chain으로 roundtrip 검증했다. 원문 stdout/전체 STATE/운영값은 복사하지 않았다.
- `research-curves.json`: CSV7개/216행과 partial/full history의 coverage limits. Qwen 보존 원본 history는 포함하되 Gemma/Liquid의 compact snapshot을 완전곡선으로 표시하지 않았다.
- `research-inventory.json`: `results/research` 파일1385개의 포함/제외·parse 실패·기존출판 중복 disposition. protected task/exposure14개는 내용 미열람, malformed metadata2개도 제외 이유 보존. raw/generated/checkpoint/비밀/운영경로는 공개하지 않는다.

공개 권장: 모델 ID/revision, cohort/split count와 manifest SHA, aggregate 수치/protocol SHA, trial 설정·validation 곡선·선택/종료 step, seed42 재사용/고정 Teacher 한계, 범위 한정 실패 분류, repo-relative source/JSON pointer/SHA provenance. 제외: corpus/토큰/prompt/answer/prediction/QA heldout, weights/checkpoints, 개별 source IDs/text-hash 목록, 절대경로/cloud resource IDs/credentials/연락처/운영로그/전체STATE. 공개모델 라이선스·gating은 별도 확인 대상이다.

메타데이터 SHA는 `evidence.json`의 `/sources_sha256`, 확대 projection 각 `/projections/*/origins/*/source_sha256`, inventory 각 `/files/*/metadata_sha256`에 있다. 해시 일치는 로컬 보존 무결성이며 외부서명·payload 재현 증명이 아니다. 부모가 선택 검증 후 반영할 준비물이며 이번 작업은 commit/push하지 않았다.
