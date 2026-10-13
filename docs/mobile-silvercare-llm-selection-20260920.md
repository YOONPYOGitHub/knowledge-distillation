# 보급형 스마트폰용 한국어 실버·의료 지원 LLM 선정 보고서

조사일: **2026-09-20**. 공식 모델 카드·라이선스·런타임 문서와 공개 파일 메타데이터를 확인했다.

> **목표:** 서버에서 학습·증류한 작은 모델을 보급형 스마트폰에 배포하여, 개인정보를 외부로 보내지 않고 한국어 돌봄 대화와 검증된 건강정보 안내를 제공한다. 스마트폰에서 대형 모델을 학습하거나, 작은 모델을 독립적인 진단·처방 주체로 만드는 것이 아니다.
>
> **증거 수준:** 모델 공개·구조·라이선스·배포 파일 크기는 확인된 사실이다. 제조사 벤치마크는 외부 보고값이다. 아래 모델 우선순위·학습값·기기별 자원 예산은 제안이며, 본 프로젝트의 폰 실측 또는 의료 안전성 검증 결과가 아니다.

## 1. 결론: 어떤 모델을 선택할 것인가

**주 연구축은 Qwen3.5-9B Teacher → 0.8B/2B Student, 텍스트 전용, non-thinking, 4-bit 배포를 권장한다. 기기가 미정인 현재는 보수적으로 4GB부터 검증한다. 4GB 경량판은 Qwen3.5-0.8B와 LFM2.5-1.2B, 6GB 품질판은 Qwen3.5-2B를 우선 후보로 둔다. 2B 단일 모델로 보급형 전체를 지원한다고 확정하지 않는다.**

| 역할·조건 | 구체적인 모델 | 권장 방향 |
|---|---|---|
| 품질 중심 Student, 우선 6GB RAM 검증 | **Qwen/Qwen3.5-2B** | post-trained 모델에 한국어 서비스 SFT·선택적 KD → GGUF Q4_K_M/Q4_0 비교 |
| 4GB RAM·CPU 중심 경쟁 Student | **LiquidAI/LFM2.5-1.2B-Instruct** | 한국어 지원·실제 모바일 자료·작은 파일이 장점. LFM 라이선스 확인을 통과하면 우선 비교 |
| Apache-2.0 경량 Student, 4GB RAM 검증 | **Qwen/Qwen3.5-0.8B** | 짧은 근거 기반 안내·일정·의도 처리 위주. 품질이 부족하면 기능 범위를 줄이며 의료 답변을 강행하지 않음 |
| 반드시 포함할 모바일 대안 | **google/gemma-4-E2B-it** 및 공식 **QAT Mobile** 변형 | LiteRT-LM으로 비교. 이름의 E2B를 실제 총 2B로 계산하지 않음 |
| 현역 대체 후보 및 비교 기준선 | **Qwen/Qwen3-1.7B**, 기존 **Qwen2.5-1.5B** | Qwen3-1.7B도 동일한 실기·안전 평가로 최종 채택 가능. 기존 2.5는 연구 기준선으로 보존 |
| 기본 Teacher, 서버 전용 | **Qwen/Qwen3.5-9B** | 근거 문서 기반 답변 생성, 같은 Qwen3.5 Student의 logit KD 후보 |
| 추가 품질 예산이 있을 때 Teacher | **Qwen/Qwen3.8-27B** | 어려운 사례의 답변 후보 생성·교차검토. 전문가 검수를 대체하지 않음 |

이 선택은 **Qwen3.5가 한국어 의료에 최고라고 입증되었기 때문이 아니다.** 공개 라이선스, 0.8B/2B/9B 규모 선택, 텍스트 전용 경로, 현재 KD 자산과의 연계성을 고려한 **우선 실험안**이다. 실기 평가에서 LFM 또는 Gemma가 승리하면 Student를 바꾸는 것이 목표에 맞다.

상용 제품의 기본안은 Apache-2.0 계열로 준비하되, **엄격한 4GB·CPU 환경에서는 LFM2.5-1.2B를 부차적인 후보로 밀어두지 않는다.** 한국어 서비스 품질과 라이선스를 통과하면 제품 기본 모델로 채택할 수 있다.

## 2. 현재 프로젝트와 제품 목표 사이의 차이

- [현재 비교 계획](korean-b-model-comparison-plan.md)은 한국어 Wikipedia, Teacher/Student, Base/FT/KD 비교와 실버케어 후속 SFT·RAG를 명시한다.
- [v3 설정](../configs/h100_qwen7b_korean_reverse_v3.yaml)은 Qwen2.5-7B/1.5B, Wikipedia 1,000문서, 길이 256, 기존 FT 체크포인트에서의 KD다.
- 확인한 문서에는 **구체적인 폰 기종·SoC, 앱 RAM 한도, 전력 예산, 모바일 export 합격 기준이 없다.** 도메인 SFT·RAG는 계획과 실제 구현을 구분해야 한다.
- [기존 GPU 결과](../results/logs/qwen7b-korean-4way-v1/notes.md)의 PPL과 tokens/sec는 스마트폰의 실시간 한국어 대화 품질·decode 속도가 아니다.

따라서 목표 함수는 **Wikipedia PPL 최소화**에서 **한국어 안전한 과업 성공률을 유지하면서 폰 메모리·지연·배터리를 줄이기**로 바뀌어야 한다. 기존 결과는 방법론 기준선으로 보존하며, 이번 조사에서는 실행 중인 학습·환경을 변경하지 않았다.

## 3. 대상 기기와 파라미터 규모

실제 지원 기기가 정해지지 않았으므로 다음은 **제품 설계 가정**이다. RAM 용량만으로 성능을 보장할 수 없으며 SoC, 메모리 대역폭, 드라이버, OS의 프로세스 종료 정책이 함께 중요하다.

| 기기 구간 | 초기 후보 범위 | 초기 앱 전체 메모리 목표 | 운용 원칙 |
|---|---|---|---|
| RAM 3GB 이하 | 범용 생성 모델 상시 실행을 기본 요구로 삼지 않음 | 별도 산정 | 검색·정형 안내·작은 분류기 중심의 기능 제한 모드 |
| RAM 4GB | **0.8~1.2B 우선**, 2B는 측정 후 | peak PSS 약 **1.5GB 이하를 목표** | CPU로도 동작, 컨텍스트 2K, 음성 모델과 동시 상주 최소화 |
| RAM 6GB | **1.2~2B 우선**, Gemma E2B 최적화판 비교 | peak PSS 약 **2.5GB 이하를 목표** | 품질 우선 Student, 2K 기본·4K 선택 |
| RAM 8GB 이상 | 2B 기본, 3~4B 확장판은 선택 | 실기 측정 | 보급형 지원의 최소 조건으로 강제하지 않음 |

PSS 목표는 실측 예측값이나 Android가 보장하는 앱 할당량이 아니다. UI, 검색 DB·인덱스, KV/recurrent state, 런타임 버퍼, 음성 처리까지 포함해 실측해야 한다.

**폰에 7B/9B를 직접 올리는 구성을 기본으로 하지 않는다.** 4-bit 순수 가중치만 대략 3.5/4.5GB이며 추가 메모리와 지연이 필요하다. 큰 모델은 학습 서버의 Teacher로 활용한다.

**KD와 양자화는 다른 작업이다.** KD는 작은 Student의 품질을 개선하는 학습이고, 4-bit는 가중치 표현을 줄이는 배포 최적화다. LoRA는 학습할 파라미터 수를 줄일 뿐 배포할 본체의 파라미터 수를 줄이지 않는다.

## 4. 후보별 조사 결과

### 4.1 Qwen3.5-2B / 0.8B — 주 실험 및 경량판

- 두 모델은 공식 post-trained 공개 모델이고 **Apache-2.0**이다. 0.8B/2B는 언어 모델 규모이며 원본에는 vision encoder도 포함된다.[S1][S2]
- 24개 언어 층 중 18개는 linear attention, 6개는 full attention이다. MoE가 아니라 **dense FFN을 쓰는 하이브리드 attention 모델**이다. 큰 Qwen 계열의 MoE 설명을 이 모델에 그대로 적용하지 않는다.
- 공식 Transformers 문서에 `Qwen3_5ForCausalLM`을 통한 **텍스트 전용 로딩**이 있다. 사진이 불필요한 1차 제품은 vision encoder/projector를 배포하지 않는 경로를 검증한다.[S3]
- 0.8B/2B는 공식 카드상 **non-thinking이 기본**이다. 두 카드 모두 thinking loop 가능성을 경고하므로, 폰 서비스에서는 명시적으로 reasoning 생성을 끄고 길이·시간 제한을 둔다.[S1][S2]
- llama.cpp는 Qwen3.5-0.8B GGUF를 공식 quickstart 예제로 사용한다. Android CPU 경로도 문서화되어 있다. 이것이 모든 보급형 GPU/NPU에서 빠르다는 의미는 아니다.[S4][S5]

**Qwen의 같은 공식 표에 있는 non-thinking 결과:**

| 지표 | Qwen3-1.7B | Qwen3.5-2B | Qwen3.5-0.8B |
|---|---:|---:|---:|
| MMLU-Pro | 40.2 | **55.3** | 29.7 |
| MMMLU | 46.7 | **56.9** | 34.1 |
| IFEval | **68.2** | 61.2 | 52.1 |

출처: [Qwen3.5-2B 공식 카드의 Language 표](https://huggingface.co/Qwen/Qwen3.5-2B#benchmark-results). **제조사 보고값이며 한국어 의료 점수가 아니다.** 일반 지식 지표는 2B가 유리하지만 지시 준수는 Qwen3-1.7B가 더 높은 항목도 있다. 따라서 최신 세대가 모든 축에서 우월하다는 주장은 하지 않는다.

판단: **2B는 품질 중심의 합리적인 시작점, 0.8B는 기능을 제한한 경량 후보**다. 0.8B에 진단·복잡한 복약 판단을 맡겨서는 안 되며, 2B 역시 독립적 의료 판단을 보장하지 않는다.

### 4.2 LFM2.5-1.2B-Instruct — 저사양 CPU 환경의 유력 경쟁안

- 실제 파라미터 **1.17B**, 16개 층(10 convolution + 6 GQA), 어휘 65,536. 한국어를 공식 지원 언어에 포함한다.[S6]
- RAG·추출·도구 사용을 권장하고, **지식 집약적인 작업은 권장하지 않는다.** 지식을 모델에 모두 암기시키기보다 로컬 근거를 읽게 하는 우리 제품 방향과 잘 맞는다.
- 공식 GGUF·ONNX가 있고, QAD(Quantization-Aware Distillation) Q4_0도 공개되어 있다. 일반 PTQ Q4_0와 별개의 체크포인트다.[S7]
- 제조사 측정: Galaxy S25 Ultra CPU, llama.cpp Q4_0, 입력 1K/출력 100토큰에서 LFM **decode 70 tok/s, memory 719MB**, Qwen3-1.7B **40 tok/s, 1306MB**. 제조사 측정 정의의 수치이며 전체 제품 앱 PSS로 해석하지 않는다.[S6]
- 이 수치는 **플래그십** 결과이고 한국어 돌봄 대화 실측도 아니다. 보급형에서 같은 수치가 나온다고 약속할 수 없다.
- **LFM Open License v1.0**이며 Apache가 아니다. 연매출 US$10M 규모에 관한 상업 이용 제한, 관계회사를 포함한 법인 정의 등을 검토해야 한다. 임계치 근처·이상 또는 해당 여부가 불명확하면 별도 계약을 확인한다.[S8]

판단: **4GB RAM·CPU 중심이라면 반드시 Qwen과 동등한 우선순위로 실기 비교**한다. 한국어 의료 서비스 평가와 라이선스 통과를 조건으로 채택한다.

### 4.3 Gemma 4 E2B — 최신 모바일 최적화 대안

- `google/gemma-4-E2B-it`은 **Apache-2.0**, 유효 2.3B / 임베딩 포함 5.1B 규모로 설명된다. E2B를 순수 2B dense 모델과 같은 파일 크기로 가정하면 안 된다.[S9]
- 공식 QAT Mobile은 선택적인 2-bit decoding layer, 최적화한 KV cache·activation을 사용하는 별도 방식이다. 일반 2-bit PTQ를 임의 적용하는 것과 다르다.[S10]
- LiteRT-LM에는 Android CPU/GPU 지원, E2B/E4B 배포, **fine-tuned safetensors에서 재변환하는 경로**가 있다.[S11]
- 공식 배포 예시 크기는 약 **2.58GB**. S26 Ultra의 CPU/GPU 속도 자료가 있지만 역시 플래그십 결과다. 다른 페이지의 정적 가중치 메모리 예상치·GPU buffer 수치를 전체 앱 RAM과 혼동하면 안 된다.
- 추가 SFT/KD 후에 원래 QAT 품질이 자동 유지되는 것은 아니다. 수정한 가중치를 실제 배포 정밀도로 재변환·재평가하고, 필요하면 QAT를 다시 수행해야 한다.

판단: **Qwen의 필수 비교 상대**다. Kotlin/LiteRT-LM 통합, 음성 입력 확장성이 중요하면 우선순위가 더 높아진다. 다만 한국어 노년층 음성 인식·대화 안전성은 별도 평가한다. 음성 입력 지원은 음성 합성 지원과 같지 않다.

### 4.4 그 밖의 후보

| 모델 | 조사 판단 |
|---|---|
| Qwen3-1.7B | Apache-2.0, text-only·llama.cpp 지원, non-thinking 가능. 3.5의 kernel/양자화 문제 또는 지시 준수 열세가 있으면 현실적인 대안.[S12] |
| EXAONE-4.0-1.2B | 한국어·온디바이스 목적에 적합한 후보. 그러나 **EXAONE 1.2-NC**는 연구·교육 중심이며 상업적 Model/Output 사용과 경쟁 모델 개발 제한이 있다. 별도 허가 없이 상용 Student 또는 다른 계열 상용 Student의 합성 데이터 Teacher로 채택하지 않는다.[S13][S14] |
| Phi-4-mini-instruct | 한국어 지원·MIT이지만 **3.8B**로 4GB 기본 기기에는 부담. 소형 후보가 과업 품질을 충족하지 못할 때 상위 기기용 비교.[S15] |
| Llama-3.2-1B-Instruct | 모바일 생태계·공식 양자화가 장점. 한국어가 공식 지원 8개 언어에 없으므로 한국어 최우선 제품의 첫 선택은 아님. 한국어 fine-tuning 자체가 불가능하다는 뜻은 아니다.[S16] |
| MedGemma-1.5-4b-it | 의료 특화 연구 후보이지만 4B이고, 공식적으로 직접 임상 판단용이 아니며 multi-turn 최적화·평가도 하지 않았다고 밝힌다. 한국어 실버 대화 Student의 자동 정답은 아니다.[S17] |
| LFM2-1.2B-Longevity | 노화 생물학·omics 연구용, 영어 모델이다. 이름에 Longevity가 있다고 노년층 한국어 돌봄 챗봇으로 선택하지 않는다.[S18] |
| Qwen3.8-Flash-Next 등 대형 MoE | 활성 파라미터가 작아도 전체 가중치·embedding 메모리가 작아지는 것은 아니다. 보급형 폰 Student에서는 제외하고 필요할 때만 서버 Teacher로 검토 |

라이선스 표시는 모델 사용·재배포 권한에 관한 검토이며 의료기기 승인이나 임상 적합성 인증이 아니다.

## 5. 실제 배포 파일 크기와 메모리 해석

2026-09-20 Hugging Face tree API에서 파일 메타데이터만 조회했다. **가중치를 다운로드하거나 폰에서 실행하지 않았다.** GB는 10억 바이트 기준이다.

| 공개 배포물 | 형식 | 조회된 파일 크기 | 출처 성격 |
|---|---|---:|---|
| Qwen3.5-0.8B | GGUF Q4_0 | **0.563GB** | [llama.cpp 조직 배포물](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF/tree/main) |
| Qwen3.5-2B | GGUF Q4_K_M | **1.281GB** | [Unsloth 변환본](https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/tree/main), Qwen 자체 배포본과 구분 |
| Qwen3.5-2B | GGUF Q4_0 | **1.215GB** | 같은 Unsloth 변환본 |
| LFM2.5-1.2B-Instruct | GGUF Q4_K_M | **0.731GB** | [LiquidAI 공식 변환본](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/tree/main) |
| LFM2.5-1.2B-Instruct | GGUF QAD Q4_0 | **0.696GB** | 같은 LiquidAI 공식 배포처 |
| Gemma-4-E2B-it | LiteRT-LM 기본 배포물 | **2.588GB** | [Google 문서 연결 배포처](https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/tree/main) |

이는 **개별 배포 파일 크기**이며 전체 앱 설치 용량·실행 RAM이 아니다. projector·음성 모델·지식 DB·업데이트용 임시 저장공간은 따로 산정한다. 형식이 다르므로 파일 크기만으로 런타임 효율 순위를 정하지 않는다.

Gemma의 2.588GB는 조회한 **기본 LiteRT-LM artifact**의 크기다. 이를 별도 QAT Mobile 변형의 최적 크기라고 단정하지 않는다. PLE·모달리티·양자화별 용량 분해는 이번 조사에서 검증하지 않았으며, 비교 실험에서는 각 후보의 실제 배포 artifact·정밀도·외부화 설정을 함께 기록한다.

$$
M_{\mathrm{runtime}} \approx M_{\mathrm{resident\ weights}} + M_{\mathrm{KV/recurrent}} + M_{\mathrm{compute\ buffers}} + M_{\mathrm{app/RAG/audio}}
$$

- 양자화 scale·고정밀 tensor 때문에 2B × 4bit / 8 = 1GB라는 계산만으로 배포 크기를 정하면 안 된다.
- 파일 mmap·GPU 복사·PLE 외부화 등 때문에 파일 크기와 실제 상주 메모리는 다르다. 메모리가 작게 보이면 page fault와 저장장치 I/O에 따른 지연도 함께 측정한다.
- Qwen3.5/LFM의 recurrent state까지 메모리에 포함한다. full attention KV cache만 측정해서 하이브리드 모델의 전체 상태 크기로 보고하지 않는다.
- **NPU TOPS 수치만으로 실행 가능·속도를 예측하지 않는다.** 지원 연산, 양자화 방식, vendor SDK와 해당 SoC를 확인한다. 1차 공통 기준은 Android ARM64 CPU다.

## 6. 어떤 버전으로 학습할 것인가: Base보다 post-trained 우선

제품 목적의 Student 초기값은 다음처럼 선택한다.

- Qwen: **Qwen3.5-2B / Qwen3.5-0.8B** — 이름에 Instruct가 없어도 공식적으로 post-trained 모델이다. `-Base`와 구분한다.
- Liquid: **LFM2.5-1.2B-Instruct**.
- Google: **gemma-4-E2B-it** 및 적절한 QAT 경로.

이유는 이미 학습된 대화 형식·지시 수행 능력을 활용해 제한된 도메인 데이터를 효율적으로 쓰기 위해서다. Base에서 시작하는 학술 대조군은 남길 수 있지만, 서비스 개발의 기본 경로로 한국어 대화 정렬을 처음부터 다시 만들 필요는 없다.

**버전을 고정하려면 모델명뿐 아니라 revision, tokenizer, chat template, 추론 엔진 commit, 양자화 방식까지 고정해야 한다.** 조사 당시 모델 revision은 부록에 기록했다. 실제 학습 시작 전에는 선택한 revision의 카드·라이선스·가중치 hash를 다시 함께 보존한다.

## 7. Teacher와 증류 방식

### 7.1 기본 Teacher: Qwen3.5-9B

9B는 폰에 배포하지 않고 H100 등 서버에서 사용한다. 텍스트 전용 BF16 가중치만 대략 18GB이며, 전체 학습 메모리는 activation·logits·Student·optimizer를 더해야 한다. 긴 컨텍스트·큰 batch를 무조건 감당한다고 가정하지 않는다.[S19]

Qwen3.5-9B/2B/0.8B 설정의 padded `vocab_size=248320`은 동일한 것을 확인했다. 그러나 **동일한 크기만으로 token ID 의미까지 같다고 보장할 수는 없다.** logit KD 전에 token→ID 전체 매핑, special token, 전처리·chat template 결과 및 logits 정렬을 확인한다.

교사의 도메인 답변 품질을 먼저 검증한다. Teacher 추가 SFT는 검수된 자료로 이득이 확인될 때 수행하고, 검증되지 않은 의료 corpus로 학습시킨 모델을 더 좋은 Teacher라고 가정하지 않는다.

### 7.2 먼저 sequence-level KD, 이후 필요할 때 logit KD

1. 사용 허가·최신성·출처가 확인된 한국어 근거 문서를 준비한다.
2. Teacher가 **근거와 질문을 입력받아 짧은 답변·구조화한 행동 라벨**을 만든다.
3. 근거 일치·수치·금기 응답을 검증하고, 의료진/돌봄 전문가 검수와 위험도별 샘플링 검수를 거친다.
4. 검수된 답변을 Student SFT 데이터로 사용한다. 이것이 **sequence-level distillation**이며 Teacher/Student tokenizer가 달라도 가능하다.
5. Qwen3.5 계열에서는 같은 조건의 SFT 대조군과 비교하면서 logit KD를 추가할지 결정한다.

LFM/Gemma Student에 Qwen Teacher를 쓸 경우 **vocabulary를 잘라 맞추거나 raw logits를 직접 KL 비교하지 않는다.** 우선 답변 기반 증류를 사용한다. 같은 계열에서도 어휘·chat template 정렬을 확인해야 한다.

대형 Teacher 옵션은 Qwen3.8-27B다.[S20] 어려운 사례 생성에 별도로 쓰되 9B보다 한국어 의료 답변이 실제로 나은지 먼저 평가한다. 교사 두 개가 동의한다고 사실이 되는 것은 아니며, 근거와 전문가 검수가 우선이다. MedGemma 보조 검토는 HAI-DEF 약관·한국어 적용 가능성을 별도 확인한 뒤에만 추가한다.

## 8. 학습·추론 파라미터 제안

아래 값은 **새 도메인 실험의 시작값**이지 이미 찾은 최적값이 아니다. 기존 Wikipedia에서 선택한 LR·alpha·T를 새 모델에 그대로 이식하지 않는다.

### 8.1 한국어 서비스 SFT

| 항목 | 시작값 / 탐색 범위 |
|---|---|
| 학습 자료 | 검수된 한국어 대화·근거 기반 QA 2만~5만 건 규모의 파일럿을 제안. 숫자보다 근거·다양성·검수 우선 |
| 최대 학습 길이 | **2048**, 메모리·대화 분포에 따라 1024/4096 비교 |
| 손실 적용 구간 | assistant 답변·필요한 행동 토큰만. system/user/근거/padding은 마스킹 |
| 기본 학습 방식 | BF16 LoRA, **rank 16, LoRA alpha 32, dropout 0.05** |
| LoRA 범위 | text backbone의 검증된 attention·MLP 선형층. 하이브리드 모델의 실제 모듈명을 확인하고 vision 영역은 제외 |
| LoRA LR | **1e-4 시작**, 5e-5 / 1e-4 / 2e-4 소규모 탐색 |
| full FT 대조군 | 필요할 때 **5e-6 / 1e-5 / 2e-5**. LoRA LR를 그대로 적용하지 않음 |
| optimizer | AdamW, weight decay 0.01, gradient clip 1.0 |
| 학습량 | 1~3 epochs 상한, validation 기준 early stopping |
| 배치 | 유효 global batch 32 예시: microbatch 1~2와 accumulation 조정. 길이가 다르면 유효 target token 예산도 기록 |
| scheduler | warmup ratio 0.03, cosine decay를 시작안으로 비교 |
| 재현성 | 고정 split, 유망 후보는 학습 seed 3개; 모델마다 동일한 원문·평가 질문 사용 |

LoRA의 `alpha=32`는 아래 KD 혼합계수 `alpha`와 다른 변수다. 명칭을 구분해 기록한다.

### 8.2 선택적 logit KD

$$
\mathcal{L}=\alpha\mathcal{L}_{CE}+(1-\alpha)T^2D_{KL}(p_T^{(T)}\parallel p_S^{(T)})
$$

- 여기서 `alpha`는 **CE 비중**, `T`는 **증류 온도**다. `alpha=1`은 CE-only 대조군이다. 프로젝트의 혼합계수 의미도 CE 비중이다.
- 시작안: **alpha 0.7, T 2**, tokenmean, assistant 유효 토큰만 적용.
- 탐색: alpha `{0.5, 0.7, 0.9, 1.0}`, T `{1, 2, 4}`. 전 조합을 무조건 실행하기보다 작은 validation 실험 후 축소한다.
- 위 식은 forward KL(Teacher || Student)이다. reverse KL(Student || Teacher)도 동일 조건으로 비교할 수 있지만, 기존 v3에서 사용했다는 이유로 의료 SFT에서도 최적이라고 간주하지 않는다.
- KD 단계의 추가 LoRA LR는 `2e-5~5e-5`부터 검증한다. SFT-only에도 **동일 추가 학습량**을 주어 증류 효과와 학습 시간 효과를 분리한다.
- 구체적으로, sequence-SFT 완료 후 validation으로 고른 **동일한 Student 가중치·adapter**를 두 군의 초기값으로 복제한다. 두 군은 같은 검수 답변 label·문서·배치 순서·target token 수·update 수를 사용하며, CE-only 추가 학습과 CE+logit KD만 다르게 한다. optimizer·seed·후보 탐색 예산도 맞추고 GPU 시간은 별도로 보고한다.
- Student·Teacher logits와 FP32 softmax/KL는 큰 메모리를 사용한다. 248K 어휘의 긴 시퀀스에서는 microbatch·sequence chunking 등 실제 구현 비용을 먼저 측정한다.

### 8.3 모바일 배포·생성

| 항목 | 시작안 |
|---|---|
| 양자화 | **4-bit 우선**. Qwen은 Q4_K_M과 CPU kernel에 유리할 수 있는 Q4_0 비교; LFM은 공식 QAD Q4_0 포함 |
| 기준 비교 | BF16/FP16 원본, Q8, Q4에 동일한 안전성·숫자·약품명 평가 적용 |
| 2/3-bit | 기본 채택하지 않음. Gemma 공식 QAT Mobile처럼 별도 학습·검증된 방식만 후보로 취급 |
| 컨텍스트 | **총 2048 tokens 기본**. system·근거·대화·최대 출력 합계로 관리; 필요 시 4096 검증 |
| 출력 | **128 tokens 기본, 256 상한**. 긴 응답 대신 2~4개 짧은 문장과 스트리밍 |
| reasoning | **명시적으로 비활성화**. 화면에서 숨기는 것만으로는 연산·대기시간이 줄지 않음 |
| 추론 batch | 1 |
| CPU threads | 2/4와 성능 코어 수 기준 비교. 모든 코어 사용을 무조건 최적으로 가정하지 않음 |
| KV cache | 먼저 FP16 등 검증된 기본값. KV 양자화는 지원되는 backend에서 품질·메모리 이득 검증 후 적용 |
| decoding | 제조사 권장값으로 기준선 측정 후 서비스 validation에서 조정. 낮은 temperature를 의료 안전 보장으로 취급하지 않음 |

제조사 non-thinking 시작값: Qwen3.5-0.8B/2B는 temperature 1.0, top_p 1.0, top_k 20, presence_penalty 2.0; LFM은 temperature 0.1, top_k 50, repetition_penalty 1.05; Gemma 4는 temperature 1.0, top_p 0.95, top_k 64.[S1][S2][S6][S9] 모델별 권장값이 다르므로 무조건 temperature 0.1로 통일하지 않는다. 정형 행동은 constrained decoding과 앱 검증으로 통제하고, 자유 응답 sampling은 별도 튜닝한다.

학습은 별도 최신 환경에서 하고, 기존 [별도 VESSL 브랜치의 호환 스택](https://github.com/YOONPYOGitHub/knowledge-distillation/blob/feature/vessl-3090x4-h100-parity/requirements-vessl.txt)과 [프로젝트 의존성](../pyproject.toml)을 실행 중에 업그레이드하지 않는다. Qwen3.5의 text loader·DeltaNet kernel, LoRA, 저장/로드, GGUF export까지 작은 end-to-end 검증을 먼저 한다. CUDA 학습 kernel 지원과 모바일 backend 지원은 별개의 문제다.

## 9. 모델에 맡길 역할과 맡기지 않을 역할

**학습시킬 것:** 존댓말·쉬운 문장·반복 질문 대응, 질문 의도와 일정 정보 추출, 검증된 문서 설명, 모르는 경우 보류, 의료진 연결 안내, 근거 없는 진단·용량 변경 요청을 거절하는 행동.

**모델 기억에 의존하지 않을 것:** 최신 약품 정보, 개인별 복용량·상호작용 판단, 응급 여부의 최종 판정, 전화번호·일정 실행 같은 정확한 시스템 동작.

권장 구조:

1. 입력 → 전문가가 설계한 안전 라우팅(규칙 + 평가된 분류기 + 필요 시 보수적 보류).
2. 일정·알림은 검증된 앱 로직으로 처리하고 사용자 확인을 받는다.
3. 건강정보는 **기기 내 검증된 지식 저장소**에서 검색한다. 근거가 없거나 오래되면 답변을 보류한다.
4. LLM은 근거를 쉬운 한국어로 설명하고 출처를 붙인다. 숫자·약품명·출처 ID를 앱이 다시 검사한다.
5. 위험 의심 또는 불확실 상황은 승인된 안내·사람 연결 경로로 전환한다. 응급 연락 UI는 모델이 실패해도 접근 가능하게 한다.

단순 키워드만으로 응급 상황을 완전히 검출할 수는 없다. 부정 표현·과거력·우회 표현·음성 오인식까지 평가한다. RAG도 잘못된 검색·오래된 문서·prompt injection 때문에 실패하므로 안전을 보장하지 않는다.

의료정보 출처는 국내 공공기관·전문기관·의료진 검수 자료에서 사용·재배포·학습 권한을 확인해 선정한다. 공개 웹페이지라고 자동으로 학습·앱 내 재배포가 허용되는 것은 아니다. 최신 지식은 문서 패키지 업데이트로 관리하고 개인 건강정보를 학습 데이터로 무단 편입하지 않는다.

음성 UI가 필요하면 오프라인 ASR/TTS까지 별도 예산에 넣는다. 네트워크를 사용하는 OS 음성 기능을 붙여 놓고 완전 오프라인이라고 표시하지 않는다. 노년층 발화·사투리·약품명·숫자의 인식 정확도는 생성 모델과 따로 평가한다.

WHO의 건강 AI 지침과 MedGemma의 공식 제한도 **사용 맥락에 맞춘 검증·감독이 필요함**을 뒷받침한다.[S17][S21] 실제 서비스 범위에 따라 국내 의료기기·개인정보 관련 전문 검토가 필요하며, 이 보고서는 임상·법률 승인이 아니다.

## 10. 최종 선정을 위한 실험 순서와 합격 기준

**학습에 투자하기 전에 export와 폰 실행 가능성을 먼저 검증한다.** 다음은 실행 계획이지 완료 사실이 아니다.

### 단계 A: 단말·배포 사전검증

- 실제 판매 지역의 **4GB 저가 기기 1대, 6GB 보급형 1대**, 비교용 상위 기기 1대를 고른다. NPU가 좋은 플래그십만으로 통과시키지 않는다.
- Qwen3.5-0.8B/2B, LFM2.5-1.2B, Gemma4 E2B, Qwen3-1.7B 및 기존 Qwen2.5 기준선을 평가한다.
- 비행기 모드에서 cold/warm load, 입력 128/512/1024 tokens, 출력 128 tokens, 2K/4K context를 측정한다. 각 모델에 동일 원문도 입력해 한국어 토큰 효율과 실제 완료 시간을 별도로 비교한다.
- 최소 30회 반복, 10분 이상 연속 대화, 배터리 구동·발열 안정화 전후를 기록한다. TTFT, prefill/decode, 한국어 문자/sec, PSS/RSS, native/GPU 메모리, 전력·배터리, 앱 강제종료를 수집한다.

### 단계 B: 도메인 데이터와 안전 평가

- 전문가 검수된 평가 500~1,000사례로 파일럿을 시작하되, 이 규모를 임상 검증으로 간주하지 않는다.
- 위험 의심, 비응급 일상, 복약 안내의 경계, 근거 없음, 오류·방언·음성 오인식, 반복 질문, 정보 부족, 악성 문서/지시를 포함한다.
- 원문 문서·사용자/환자·시나리오 템플릿 단위로 split하여 비슷한 합성 질문이 train/test에 동시에 섞이지 않게 한다.
- 위험 사례 누락률·과잉 경고율, 근거 일치, 숫자/단위/약품명 보존, 거절/보류 적절성, 노년층 이해 가능성을 별도로 보고한다. KMMLU·KoBEST는 보조 지표이지 의료 안전성 인증이 아니다.

### 단계 C: 학습 효과 분리

1. 원본 post-trained + 동일 RAG.
2. 검수된 사람 작성 데이터로 SFT + 동일 RAG.
3. 같은 사람 데이터에 검수된 Teacher 답변을 추가한 SFT(sequence KD) + 동일 RAG.
4. 같은 자료·추가 학습량의 SFT 대조군과 logit KD 비교(호환되는 조합만).
5. 각 후보의 최종 Q4/모바일 QAT 배포본으로 **동일 평가를 다시 수행**.

같은 tokenizer 내 PPL은 진단용으로 쓰되, 다른 tokenizer의 PPL을 직접 순위화하지 않는다. 주 지표는 실제 과업·안전성·기기 자원이다. validation으로 선택하고 최종 test는 freeze 후 평가한다.

### 초기 엔지니어링 목표 — 실측값·임상 승인 기준 아님

- 입력 512 tokens에서 warm TTFT **p95 3초 이내**, 지속 decode **10 tokens/sec 이상**을 초기 목표로 둔다. 한국어 문장 완료 시간도 함께 제한한다.
- 앱 메모리는 3절의 기기별 목표를 충족하고, 연속 사용 중 OOM·OS 강제종료가 없어야 한다.
- 명시적인 위험 시나리오의 필수 안내 누락, 근거 없는 진단/복용량 생성 등 중대한 실패는 원인 수정 전 출시 차단 사유다.
- 개발 단계 차단 규칙으로, 사전에 정의한 치명적 오류가 고정된 sentinel 평가에서 **1건 이상이면 실패**로 제안한다. 예상 행동과 치명성 정의는 의료진이 작성하고, 고위험 사례는 두 명이 독립 평가한 뒤 불일치를 조정한다. 일반 과업 성공률·민감도·과잉 경고율의 허용치와 신뢰구간은 임상 책임자가 용도별로 승인해야 하며, 이 보고서에서 임의의 수치를 임상 합격선으로 만들지 않는다.
- 안전 평가의 표본 수·신뢰구간·하위집단 결과를 제시한다. 테스트에서 실패 0건이어도 현실 위험이 0이라는 뜻은 아니다.
- 양자화 후 치명적 오류가 증가하면 Q5/Q8, 다른 모델 또는 기능 제한을 선택한다. 작은 파일을 위해 안전성 열화를 수용하지 않는다.

**안전·라이선스·실기 제약을 먼저 통과시킨 후, 통과 모델 사이에서 품질/속도/메모리/배터리의 균형을 선택한다.** 모든 후보가 실패하면 모델을 억지로 확정하지 않고 서비스 범위를 축소하거나 지원 기기 기준을 조정한다.

## 11. 실행 의사결정 요약

1. 기존 Qwen2.5 실험은 보존한다. 더 이상 Wikipedia PPL만으로 최종 폰 모델을 결정하지 않는다.
2. **Qwen3.5-2B와 Qwen3.5-0.8B의 post-trained 버전**을 주 연구축으로 잡는다.
3. **LFM2.5-1.2B-Instruct와 Gemma4 E2B 모바일 버전**을 같은 데이터·실기에서 반드시 비교한다.
4. **Qwen3.5-9B Teacher → 검수된 한국어 답변 → Student SFT → 선택적 KD → 4-bit → 폰 재평가** 순서로 진행한다.
5. 4GB 기기의 주 모델은 LFM1.2B 또는 Qwen0.8B에서 먼저 찾고, 6GB 품질판은 Qwen2B를 먼저 검증한다. Gemma 모바일 최적화판이 해당 기기에서 더 좋으면 이를 채택한다.
6. 최종 제품은 **작은 LLM + 로컬 근거 검색 + 앱의 안전 로직 + 사람 연결**이다. 모델 하나에 의료 서비스 전체를 맡기지 않는다.

## 부록 A. 조사 당시 revision

Hugging Face 모델 API에서 확인한 값이다. 실제 학습/배포 manifest에는 tokenizer·라이선스·변환본·엔진의 별도 revision/hash도 추가한다.

| 모델 | revision |
|---|---|
| Qwen/Qwen3.5-0.8B | `2fc06364715b967f1860aea9cf38778875588b17` |
| Qwen/Qwen3.5-2B | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Qwen/Qwen3.5-9B | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| LiquidAI/LFM2.5-1.2B-Instruct | `0f604ada3f766f9f257460c4c9f0b5d6f69d431b` |
| google/gemma-4-E2B-it | `3e22461f65e89153144f8adb70e3b8c2cc9845a7` |
| google/gemma-4-E2B-it-qat-mobile-transformers | `dd693ff40353f057ca5f07e945ad867f4afbf2ec` |

## 부록 B. 출처

아래는 2026-09-20 열람한 공식 1차 자료다. 별도 표시한 Unsloth·LiteRT community·ggml-org 배포 파일은 원 개발사 가중치와 변환 주체를 구분했다. 모델 카드의 벤치마크·권장값은 서로 다른 측정 조건일 수 있으며, 이 보고서의 실험 제안값과 구별한다.

- [S1 — Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [S2 — Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- [S3 — Transformers Qwen3.5 text-only·hybrid 문서](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [S4 — llama.cpp 공식 프로젝트](https://github.com/ggml-org/llama.cpp)
- [S5 — llama.cpp Android 문서](https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md)
- [S6 — LFM2.5-1.2B-Instruct 모델·한국어·모바일 측정](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- [S7 — LFM2.5 공식 GGUF·QAD 배포](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF)
- [S8 — LFM Open License v1.0 원문](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/blob/main/LICENSE)
- [S9 — Gemma 4 E2B 공식 모델 카드](https://huggingface.co/google/gemma-4-E2B-it)
- [S10 — Gemma 4 E2B QAT Mobile](https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers)
- [S11 — LiteRT-LM Gemma 4: 배포·fine-tuned 모델 변환·기기 측정](https://developers.google.com/edge/litert-lm/models/gemma-4)
- [S12 — Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [S13 — EXAONE-4.0-1.2B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B)
- [S14 — EXAONE 1.2-NC 원문](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-1.2B/blob/main/LICENSE)
- [S15 — Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [S16 — Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [S17 — Google MedGemma 1.5: intended use・limitations](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card)
- [S18 — LFM2-1.2B-Longevity](https://huggingface.co/LiquidAI/LFM2-1.2B-Longevity)
- [S19 — Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [S20 — Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- [S21 — WHO: Ethics and governance of AI for health, LMM guidance](https://www.who.int/publications/i/item/9789240084759)