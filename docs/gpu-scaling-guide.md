# GPU 환경별 KD 실험 가이드

> GPU 사양별로 적합한 Teacher/Student 모델, 학습 데이터, 평가 기준을 정리한 실전 가이드.
> 현재 로컬(MPS) 기반 [exp01~exp08](../configs/) 실험을 GPU 환경으로 확장할 때 참조.

---

## 목차

1. [핵심 원칙](#1-핵심-원칙)
2. [GPU 티어별 추천 조합](#2-gpu-티어별-추천-조합)
3. [데이터셋 카탈로그](#3-데이터셋-카탈로그)
4. [평가 기준 (5-Tier)](#4-평가-기준-5-tier)
5. [티어별 추천 패키지](#5-티어별-추천-패키지)
6. [의사결정 트리](#6-의사결정-트리)
7. [구현 시 주의사항](#7-구현-시-주의사항)

---

## 1. 핵심 원칙

KD 조합 선택 시 반드시 지켜야 할 3가지:

1. **같은 tokenizer / vocab** — logits 차원이 일치해야 KL loss 직접 계산 가능
2. **적절한 capacity gap** — Teacher가 Student보다 **1.5x ~ 8x** 크기가 sweet spot
3. **같은 모델 패밀리** — hidden state matching, attention transfer 등 확장 가능

> ⚠️ **Chinchilla 스케일링 법칙**: 모델 1B 파라미터당 학습 토큰 약 20B 권장.
> 단, KD는 Teacher의 soft label 정보 덕분에 더 적은 토큰으로도 학습 가능 (보통 5~10B / 1B params).

---

## 2. GPU 티어별 추천 조합

### Tier 0: 로컬 (MPS / CPU) — 현재 프로젝트 baseline

| 항목 | 값 |
|---|---|
| **Teacher** | `gpt2` (124M) |
| **Student** | `distilgpt2` (82M) |
| **데이터셋** | WikiText-2 (2M tokens) |
| **Batch / SeqLen** | 4 / 256 |
| **목적** | 코드 동작 검증, KD 효과 시그널 확인 |
| **소요 시간** | ~1시간 (M1/M2 MPS) |
| **참고 config** | [exp06_teacher_ft_smoke.yaml](../configs/exp06_teacher_ft_smoke.yaml) |

---

### Tier A: 단일 24GB GPU (RTX 4090 / A5000 / L4)

| 항목 | 권장 | 비고 |
|---|---|---|
| **Teacher** | `meta-llama/Llama-3.2-3B` | FP16 ~6GB |
| **Student** | `meta-llama/Llama-3.2-1B` | FP16 ~2GB |
| **비율** | 3x | Sweet spot |
| **데이터셋** | OpenWebText subset 또는 WikiText-103 | 100M~1B tokens |
| **학습 토큰 수** | **1B~3B tokens** | KD라 Chinchilla보다 적게 가능 |
| **Batch size** | 8 (effective 32 with grad accum 4) | seq_len 1024 |
| **Epochs** | 1~3 | LLM은 1 epoch도 충분한 경우 많음 |
| **Precision** | BF16 권장 | |
| **예상 학습 시간** | ~24시간 (1B tokens, 4090 기준) | |

**대안 조합**:
- `Pythia-1.4B → Pythia-410M` + WikiText-103 (~12시간) — 연구용 깔끔한 비교
- `Qwen2.5-1.5B → Qwen2.5-0.5B` + 도메인 특화 데이터 — 한국어 강함

---

### Tier B: 단일 40~80GB GPU (A100 / H100)

| 항목 | 권장 | 비고 |
|---|---|---|
| **Teacher** | `meta-llama/Llama-3.1-8B` | BF16 ~16GB |
| **Student** | `meta-llama/Llama-3.2-1B` 또는 `Llama-3.2-3B` | |
| **비율** | 2.7x ~ 8x | |
| **데이터셋** | **SlimPajama** subset 또는 **FineWeb-Edu** | 10B~50B tokens |
| **학습 토큰 수** | **5B~20B tokens** | |
| **Batch size** | 16 (effective 64~128) | seq_len 2048 |
| **Epochs** | 1 | |
| **추가 기법** | **Teacher logits pre-compute & cache** | Teacher forward 비용 절감 핵심 |
| **예상 학습 시간** | ~3~5일 (5B tokens, A100 80GB) | |

> 💡 Teacher logits 캐싱: top-K (예: 50) logits만 디스크에 저장하면 5~10x 속도 향상.

---

### Tier C: 멀티 GPU (4~8 × A100 80GB)

| 항목 | 권장 |
|---|---|
| **Teacher** | `Qwen2.5-32B` 또는 `Llama-3.1-70B` (4-bit 양자화 로드) |
| **Student** | `Qwen2.5-7B` 또는 `Llama-3.1-8B` |
| **비율** | 4.6x ~ 8.75x |
| **데이터셋** | FineWeb (15TB) subset, RedPajama-v2, RefinedWeb |
| **학습 토큰 수** | **50B~200B tokens** |
| **분산 학습** | DeepSpeed ZeRO-3 또는 PyTorch FSDP |
| **Batch size** | 32 per GPU × 4~8 GPUs |
| **예상 학습 시간** | ~1~2주 (50B tokens) |

---

### 티어별 한눈에 보기

| Tier | GPU | Teacher | Student | 데이터 토큰 | 시간 |
|---|---|---|---|---|---|
| **0** | MPS/CPU | gpt2 (124M) | distilgpt2 (82M) | 2M | 1h |
| **A** | 24GB | Llama-3.2-3B | Llama-3.2-1B | 1B | ~1d |
| **B** | 80GB | Llama-3.1-8B | Llama-3.2-1B/3B | 10B | ~3-5d |
| **C** | 4-8×80GB | Qwen2.5-32B / Llama-3.1-70B | Qwen2.5-7B / Llama-3.1-8B | 50B+ | ~1-2w |

---

## 3. 데이터셋 카탈로그

### 일반 코퍼스 (Pretraining KD)

| 데이터셋 | 크기 | 토큰 수 | 적합 모델 | 라이선스 | HF ID |
|---|---|---|---|---|---|
| WikiText-2 | 12MB | 2M | <500M | CC-BY-SA | `wikitext` (`wikitext-2-raw-v1`) |
| **WikiText-103** | 500MB | 100M | 500M~1B | CC-BY-SA | `wikitext` (`wikitext-103-raw-v1`) |
| **OpenWebText** | 38GB | 8B | 1B~3B | 공개 | `Skylion007/openwebtext` |
| C4 (cleaned) | 750GB | 156B | 3B+ | ODC-BY | `allenai/c4` |
| The Pile | 825GB | 300B | 3B+ | 다양 | `EleutherAI/pile` |
| **SlimPajama** | 627GB | 627B | 7B+ | Apache 2.0 | `cerebras/SlimPajama-627B` |
| **FineWeb** | 15TB | 15T | 7B+ | ODC-BY | `HuggingFaceFW/fineweb` |
| **FineWeb-Edu** | 1.3TB | 1.3T | 1B~7B | ODC-BY | `HuggingFaceFW/fineweb-edu` |
| RedPajama-v2 | 30TB | 30T | 본격 LLM | Apache 2.0 | `togethercomputer/RedPajama-Data-V2` |

### 도메인 특화

| 도메인 | 데이터셋 |
|---|---|
| 코드 | `bigcode/the-stack-v2`, `bigcode/starcoderdata` |
| 한국어 | `HAERAE-HUB/KoBEST_v1`, AI Hub 한국어 코퍼스, `allenai/c4` (`ko`) |
| 의학 | `pubmed`, MIMIC-III/IV |
| 법률 | `pile-of-law/pile-of-law` |
| 수학 | `nvidia/OpenMathInstruct-1`, `EleutherAI/proof-pile-2` |

### Instruction Tuning (SFT KD)

| 데이터셋 | 샘플 수 | 용도 |
|---|---|---|
| `tatsu-lab/alpaca` | 52K | 기본 instruction |
| `Open-Orca/OpenOrca` | 4M | GPT-4 trace 풍부 |
| `HuggingFaceH4/ultrachat_200k` | 200K | 다회차 대화 |
| `argilla/ultrafeedback-binarized` | 64K | DPO/RLHF용 |

---

## 4. 평가 기준 (5-Tier)

GPU 환경에서는 평가도 풍부하게 할 수 있으니, 5단계 체계로 구성.

### 🥇 Tier 1: 핵심 지표 (반드시 측정)

| 지표 | 측정 대상 | 비고 |
|---|---|---|
| **Perplexity** | Held-out 데이터 | 현재 [evaluate.py](../src/evaluate.py)에 구현됨 |
| **Bits-per-Byte (BPB)** | Tokenizer-독립 PPL | 모델 간 비교 시 필수 |
| **Top-1 / Top-5 Accuracy** | 토큰 예측 정확도 | PPL 보완 |
| **KL(Student ‖ Teacher)** | KD의 직접 효과 | KD 본질 측정 |
| **Top-1 Agreement Rate** | Student-Teacher argmax 일치율 | 직관적 지표 |

### 🥈 Tier 2: 표준 LLM 벤치마크 (lm-evaluation-harness)

```bash
pip install lm-eval
lm_eval --model hf --model_args pretrained=path/to/student \
        --tasks hellaswag,piqa,winogrande,arc_easy,arc_challenge,lambada_openai,boolq,sciq \
        --device cuda:0 --batch_size 16
```

| 벤치마크 | 측정 능력 | 1B 모델 baseline |
|---|---|---|
| **LAMBADA** | 장문 이해 (마지막 단어 예측) | ~50% |
| **HellaSwag** | 상식 추론 (문장 완성) | ~45% |
| **PIQA** | 물리 상식 | ~70% |
| **WinoGrande** | 대명사 참조 | ~58% |
| **ARC-Easy** | 초등 과학 | ~55% |
| **ARC-Challenge** | 어려운 과학 | ~30% |
| **OpenBookQA** | 오픈북 QA | ~32% |
| **BoolQ** | 예/아니오 QA | ~62% |
| **TriviaQA** | 사실 지식 | ~25% |
| **SciQ** | 과학 QA | ~85% |

> ⚠️ MMLU, GSM8K는 7B+ 모델용. 1B 모델에서는 random에 가까움.

### 🥉 Tier 3: 생성 품질 (KD 논문 표준)

WikiText 또는 CNN-DailyMail prefix → 50~100 토큰 generation → 참조와 비교.

| 지표 | 도구 |
|---|---|
| **ROUGE-1/2/L** | `rouge-score` |
| **BLEU** | `sacrebleu` |
| **BERTScore** | `bert-score` |
| **Distinct-1/2** | 자체 구현 (생성 다양성) |
| **MAUVE** | `mauve-text` (분포 수준 인간성) |
| **Self-BLEU** | 다양성의 역지표 |

### 🏆 Tier 4: 효율성 지표 (서비스 관점)

| 지표 | 측정 방식 |
|---|---|
| **Throughput @ batch=1, 8, 32, 64** | tokens/sec |
| **Latency p50 / p95 / p99** | ms |
| **First Token Latency (TTFT)** | 챗봇 UX 핵심 |
| **Peak VRAM** | `torch.cuda.max_memory_allocated()` |
| **Model size on disk** | MB |
| **FLOPs per token** | 이론적 계산량 |
| **Energy / token (Wh)** | NVIDIA-SMI 연동 (옵션) |

### 🎤 Tier 5: 주관 평가 (Instruction model 한정)

| 지표 | 도구 |
|---|---|
| **MT-Bench** | GPT-4 judge, 80개 다회차 질문 |
| **AlpacaEval 2.0** | GPT-4 pairwise win rate |
| **Arena (LMSYS)** | 인간 평가 (외부 의뢰) |
| **LLM-as-Judge (자체)** | Claude/GPT-4 활용 |

---

## 5. 티어별 추천 패키지

### 패키지 1: "단일 24GB GPU 표준 실험" ⭐ 추천 시작점

```yaml
# configs/exp10_llama32_3b_to_1b.yaml (제안)
teacher_model: meta-llama/Llama-3.2-3B
student_model: meta-llama/Llama-3.2-1B

dataset_name: Skylion007/openwebtext  # 또는 wikitext-103
max_seq_length: 1024
batch_size: 8
gradient_accumulation_steps: 4

teacher_epochs: 1
teacher_learning_rate: 1.0e-5

epochs: 1
learning_rate: 5.0e-5
temperature: 2.0
alpha: 0.3

bf16: true
target_tokens: 1_000_000_000  # 1B 토큰
```

**평가 구성**: Tier 1 + Tier 2 (6개 task) + Tier 4 throughput

**예상 성과**: KD Student가 baseline FT 대비 PPL 5~10% 개선, HellaSwag/PIQA 등에서 +2~5%p

---

### 패키지 2: "A100 본격 실험"

```yaml
teacher_model: meta-llama/Llama-3.1-8B
student_model: meta-llama/Llama-3.2-1B

dataset_name: HuggingFaceFW/fineweb-edu
dataset_subset: sample-10BT
max_seq_length: 2048
batch_size: 16
gradient_accumulation_steps: 4

teacher_logits_cache: true   # 핵심: top-K logits 디스크 저장
top_k_logits: 50

epochs: 1
learning_rate: 3.0e-5
temperature: 2.0
alpha: 0.5

bf16: true
target_tokens: 10_000_000_000  # 10B 토큰
```

**평가 구성**: Tier 1 + Tier 2 (전체) + Tier 3 + Tier 4

---

### 패키지 3: "Instruction-tuned KD (실용 결과물)"

3단계 파이프라인:

```
Stage 1: Pretraining KD       (위 패키지 1 또는 2)
Stage 2: SFT KD               on UltraChat / OpenOrca
Stage 3: DPO Distillation     on UltraFeedback
```

**평가 추가**: Tier 5 (MT-Bench, AlpacaEval)

---

### 패키지 4: "순수 연구 — Pythia 깔끔한 비교"

```yaml
teacher_model: EleutherAI/pythia-2.8b
student_model: EleutherAI/pythia-410m

dataset_name: wikitext
dataset_config: wikitext-103-raw-v1
```

> ✅ Pythia는 모든 사이즈가 **완전히 동일한 데이터/순서**로 학습됨 → 순수 capacity 효과 분리 가능. 논문/연구 발표용 최적.

---

## 6. 의사결정 트리

```
GPU 한 대만 있다?
├─ 24GB → [패키지 1] Llama-3.2-3B → 1B + OpenWebText 1B 토큰
└─ 80GB → [패키지 2] Llama-3.1-8B → 1B + FineWeb-Edu 10B 토큰

여러 대 있다?
└─ [Tier C]  Qwen2.5-32B → 7B + FineWeb 50B 토큰

서비스 만들 거다?
└─ [패키지 3] 위에 SFT + DPO 단계 추가

순수 연구 (논문 발표)?
└─ [패키지 4] Pythia-2.8B → 410M (같은 데이터 학습 → 깔끔한 비교)
```

---

## 7. 구현 시 주의사항

### 7.1 Tokenizer 호환성 검증 (필수)

```python
from transformers import AutoTokenizer

t1 = AutoTokenizer.from_pretrained(teacher_model)
t2 = AutoTokenizer.from_pretrained(student_model)
assert t1.vocab_size == t2.vocab_size, "Vocab mismatch — KL loss 계산 불가"
```

### 7.2 Teacher Logits 캐싱 전략 (대용량 데이터 필수)

| 방식 | 장점 | 단점 | 적용 |
|---|---|---|---|
| **온라인** | 코드 단순 | 매 step Teacher forward → 2~3x 느림 | <1B tokens |
| **캐싱 (top-K)** | 5~10x 속도 향상 | 디스크 용량 (~수십~수백 GB) | 5B+ tokens |

캐싱 시 권장 설정:
- `top_k_logits: 50` — 정보 손실 거의 없음
- 저장 포맷: `np.float16` 또는 `bfloat16`
- 데이터 충돌 방지를 위해 데이터셋 hash + tokenizer hash로 캐시 키 생성

### 7.3 Mixed Precision

- **BF16** 권장 (A100/H100 지원, FP16보다 안정적)
- 24GB GPU에서는 FP16도 가능하나 loss scaling 필요
- 현재 [config.py](../src/config.py)의 `fp16` 옵션을 `bf16` 추가 지원 필요

### 7.4 Gradient Accumulation

큰 모델은 batch size가 작아짐. effective batch size 유지를 위해 accumulation 사용:

```python
effective_batch = batch_size × gradient_accumulation_steps × num_gpus
# 권장: 64~256
```

### 7.5 분산 학습

- 7B+ 모델은 단일 GPU에 안 들어감 → DDP/FSDP/DeepSpeed 필수
- 현재 코드는 single-GPU 가정 → 확장 시 [src/distill.py](../src/distill.py) 리팩토링 필요
- 가장 쉬운 경로: HuggingFace `accelerate` + `DeepSpeed ZeRO-3`

### 7.6 평가 비용 관리

- Tier 2 (lm-eval-harness): 1B 모델 기준 8개 task 약 30분~1시간
- Tier 5 (GPT-4 judge): MT-Bench 80개 × 모델 4개 = ~$50~100 API 비용
- **CI에서는 Tier 1만**, 본 실험 종료 시 Tier 2~4 일괄 실행 권장

---

## 8. 즉시 시작 추천

가장 ROI 높은 첫 GPU 실험:

> **`Llama-3.2-3B (Teacher) → Llama-3.2-1B (Student)` + OpenWebText 1B tokens + Tier 1+2 평가**

이유:
1. Llama 3.2는 Meta가 **공식적으로 1B/3B를 작은 디바이스용으로 KD한 라인업** — 논문 사례 있음
2. Tokenizer 100% 호환
3. 단일 24GB GPU에서 가능 (Azure NC-T4, Colab Pro+ A100 등)
4. 결과물(1B 모델)이 **모바일/엣지 배포 가능한 실용적 사이즈**
5. lm-eval-harness로 표준 비교 가능

---

## 관련 문서

- [model-selection-guide.md](model-selection-guide.md) — 모델 선택 일반론
- [model-installation-guide.md](model-installation-guide.md) — 모델 설치/로드 가이드
- [papers.md](papers.md) — 참고 논문 목록 (KD 평가 지표 매핑 포함)
- [research-direction-guide.md](research-direction-guide.md) — 연구 방향
- [schedule.md](schedule.md) — 실험 일정

---

> **마지막 업데이트:** 2026-05-12
> **관리 기준:** GPU 환경으로 확장하는 시점에 갱신
