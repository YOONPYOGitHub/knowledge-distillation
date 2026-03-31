# 참고 논문 & 레퍼런스 관리

> 본 프로젝트에서 참조하는 논문·기술 자료 목록  
> 각 논문에서 어떤 개념을 차용했는지, 프로젝트와의 관계를 명시

---

## 목차

1. [핵심 논문 (Core Papers)](#1-핵심-논문-core-papers)
2. [LLM 특화 증류 논문](#2-llm-특화-증류-논문)
3. [최신 논문 (2024~2026)](#3-최신-논문-20242026)
4. [실전 적용 & 경량화 관련](#4-실전-적용--경량화-관련)
5. [논문 → 프로젝트 매핑](#5-논문--프로젝트-매핑)
6. [읽기 순서 추천](#6-읽기-순서-추천)

---

## 1. 핵심 논문 (Core Papers)

### 1.1 ⭐ Distilling the Knowledge in a Neural Network

> **이 프로젝트의 근간이 되는 원조 논문**

- **저자:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean
- **연도:** 2015
- **링크:** https://arxiv.org/abs/1503.02531
- **핵심 기여:**
  - Knowledge Distillation (KD) 개념 최초 정립
  - Temperature scaling을 통한 soft label 생성
  - KD Loss = KL Divergence 기반 설계
- **본 프로젝트에서 차용한 것:**
  - ✅ **전체 KD Loss 수식**: `L = α·CE + (1-α)·T²·KL`
  - ✅ **Temperature 개념**: soft label 생성 방식
  - ✅ **Teacher-Student 프레임워크** 전체 구조
  - 📌 4-Way 비교 구조는 DistilBERT (2019), Zephyr (2023)에서 차용 (아래 섹션 5 참조)

### 1.2 ⭐ DistilBERT, a distilled version of BERT

> **NLP Transformer에 KD를 적용한 대표 실전 논문**

- **저자:** Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf (Hugging Face)
- **연도:** 2019
- **링크:** https://arxiv.org/abs/1910.01108
- **핵심 기여:**
  - BERT를 40% 축소하면서 97% 성능 유지
  - Triple loss: KD loss + MLM loss + Cosine embedding loss
  - 사전학습 단계에서의 증류 (pre-training distillation)
- **본 프로젝트에서 차용한 것:**
  - ✅ **NLP 모델 증류 실험 설계 방법론**
  - ✅ **성능 비교 테이블 구성** (모델별 지표 나란히 비교)
  - 📌 참고: 우리는 fine-tuning 단계 증류만 수행 (pre-training 증류는 미포함)

### 1.3 TinyBERT: Distilling BERT for Natural Language Understanding

> **레이어별 증류(Layer-wise Distillation) 기법**

- **저자:** Xiaoqi Jiao, Yichun Yin, Lifeng Shang, et al.
- **연도:** 2020
- **링크:** https://arxiv.org/abs/1909.10351
- **핵심 기여:**
  - Transformer의 attention, hidden state, embedding 각 레이어별 증류
  - 2-stage distillation: general → task-specific
  - BERT-base 대비 7.5x 작으면서 96% 성능
- **본 프로젝트에서 차용한 것:**
  - 📌 참고만: 본 프로젝트는 logit-level 증류만 수행 (layer-wise는 확장 과제)
  - ✅ **2-stage 학습 개념** 참고 (향후 확장 시)

### 1.4 MiniLM: Deep Self-Attention Distillation

- **저자:** Wenhui Wang, Furu Wei, Li Dong, et al. (Microsoft)
- **연도:** 2020
- **링크:** https://arxiv.org/abs/2002.10957
- **핵심 기여:**
  - Self-attention 분포를 증류 (value-relation transfer)
  - Teacher의 마지막 레이어 attention만 증류해도 효과적
- **본 프로젝트에서 차용한 것:**
  - 📌 참고만: attention distillation은 미구현, logit-level만 사용

---

## 2. LLM 특화 증류 논문

### 2.1 ⭐ GKD: Generalized Knowledge Distillation for Auto-Regressive Sequence Models

> **LLM 증류의 핵심 문제를 해결한 중요 논문**

- **저자:** Rishabh Agarwal, Nino Vieillard, et al. (Google DeepMind)
- **연도:** 2023
- **링크:** https://arxiv.org/abs/2306.13649
- **핵심 기여:**
  - 기존 KD의 **train-inference mismatch** 문제 지적
    - 학습 시: Teacher 토큰 기반으로 학습
    - 추론 시: Student 자기 자신이 생성한 토큰 기반 → 분포 불일치
  - **On-policy distillation**: Student가 생성한 시퀀스에 대해 Teacher 분포를 학습
  - Reverse KL, Forward KL, JSD 등 다양한 divergence 비교
- **본 프로젝트에서 차용한 것:**
  - ✅ **Auto-regressive 모델 증류 시 주의점** 인식
  - 📌 기본 구현은 off-policy (표준 KD), on-policy는 확장 과제로 고려

### 2.2 ⭐ MiniLLM: Knowledge Distillation of Large Language Models

> **LLM에 특화된 KD 방법론**

- **저자:** Yuxian Gu, Li Dong, Furu Wei, Minlie Huang
- **연도:** 2024
- **링크:** https://arxiv.org/abs/2306.08543
- **핵심 기여:**
  - Forward KL (표준)이 LLM에서는 부적합할 수 있음을 증명
  - **Reverse KL divergence** 사용으로 Student의 mode-seeking 유도
  - Policy gradient 기반 최적화로 generation quality 향상
  - GPT-2, OPT, LLaMA 계열에서 실험
- **본 프로젝트에서 차용한 것:**
  - ✅ **GPT-2 계열 실험 설계 참조** (동일 모델 패밀리 사용)
  - ✅ **Forward KL vs Reverse KL 비교** 관점 인식
  - 📌 기본 구현은 Forward KL, Reverse KL은 확장 과제

### 2.3 ⭐ Zephyr: Direct Distilled LLM Alignment

> **LLM alignment에 증류를 적용한 최신 접근**

- **저자:** Lewis Tunstall, Edward Beeching, et al. (Hugging Face)
- **연도:** 2023
- **링크:** https://arxiv.org/abs/2310.16944
- **핵심 기여:**
  - dSFT (distilled Supervised Fine-Tuning) + dDPO (distilled DPO) 2단계
  - Teacher(GPT-4 등)의 출력 데이터로 Student(Mistral-7B) 학습
  - **Black-box distillation**: Teacher logit 없이 생성 텍스트만으로 증류
  - 7B 모델이 70B 모델 수준 성능 달성
- **본 프로젝트에서 차용한 것:**
  - ✅ **4-Way 비교 구조**: Teacher / KD / SFT / Base 비교 방식 참조
  - 📌 White-box(logit 접근 가능) 증류 수행 — Zephyr의 black-box와는 다른 접근

### 2.4 Lion: Adversarial Distillation of Closed-Source LLM

- **저자:** Yuxin Jiang, Chunkit Chan, Mingyang Chen, Wei Wang
- **연도:** 2023
- **링크:** https://arxiv.org/abs/2305.12870
- **핵심 기여:**
  - Closed-source Teacher (ChatGPT)로부터 adversarial 방식으로 증류
  - Referee 모델이 Teacher와 Student 출력 차이를 판별
  - 어려운 instruction을 자동 생성하여 증류 효과 극대화
- **본 프로젝트에서 차용한 것:**
  - 📌 참고만: 우리는 open-source white-box 증류 수행

---

## 3. 최신 논문 (2024~2026)

### 3.1 ⭐ DistiLLM: Towards Streamlined Distillation for Large Language Models

- **저자:** Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun
- **연도:** 2024
- **링크:** https://arxiv.org/abs/2402.03898
- **핵심 기여:**
  - Skew KL Divergence 도입 — Forward KL과 Reverse KL 사이의 절충
  - Adaptive off-policy 접근: Student 생성 시퀀스 활용하면서도 효율적
  - GPT-2, OPT, LLaMA 계열에서 기존 KD 대비 성능 향상 입증
- **본 프로젝트에서 차용한 것:**
  - ✅ **GPT-2 실험 벤치마크 비교 기준** 참조
  - 📌 Skew KL은 확장 과제로 고려

### 3.2 ⭐ Distilling Step-by-Step

- **저자:** Cheng-Yu Hsieh, Chun-Liang Li, Chih-Kuan Yeh, et al. (Google)
- **연도:** 2023
- **링크:** https://arxiv.org/abs/2305.02301
- **핵심 기여:**
  - Teacher의 **reasoning (chain-of-thought)**을 Student에게 증류
  - Label + Rationale 동시 학습
  - 770x 작은 모델이 few-shot LLM 성능 초과
- **본 프로젝트에서 차용한 것:**
  - 📌 참고만: 본 프로젝트는 logit-level 증류에 집중 (CoT 증류는 확장)

### 3.3 Knowledge Distillation for Closed-Source Language Models

- **저자:** Various (2024 survey papers)
- **연도:** 2024
- **링크:** https://arxiv.org/abs/2401.07013
- **핵심 기여:**
  - Black-box vs White-box KD 분류 체계 정리
  - API-only Teacher (GPT-4, Claude 등) 활용 전략
  - Data augmentation을 통한 증류 데이터 확보 방법

### 3.4 ⭐ MiniCPM: Unveiling the Potential of Small Language Models

- **저자:** Shengding Hu, Yuge Tu, et al. (Tsinghua, ModelBest)
- **연도:** 2024
- **링크:** https://arxiv.org/abs/2404.06395
- **핵심 기여:**
  - 2.4B 파라미터로 7B~13B 수준 성능 달성
  - Model Wind Tunnel 실험 방법론 — 소규모로 최적 설정을 찾고 확대
  - WSD (Warmup-Stable-Decay) 학습률 스케줄러
- **본 프로젝트에서 차용한 것:**
  - ✅ **소형 모델의 잠재력** 관점 — 증류로 어디까지 끌어올릴 수 있는가

### 3.5 ⭐ Phi-4 / Phi-3 Series (Microsoft)

- **저자:** Microsoft Research
- **연도:** 2024~2025
- **링크:** https://arxiv.org/abs/2404.14219 (Phi-3), https://arxiv.org/abs/2412.08905 (Phi-4)
- **핵심 기여:**
  - 고품질 합성 데이터로 소형 모델의 성능을 극대화
  - "Data quality > Data quantity" 입증
  - 3.8B 모델이 7B~13B 모델과 경쟁
- **본 프로젝트에서 차용한 것:**
  - ✅ **데이터 품질의 중요성** 인식
  - 📌 합성 데이터 생성은 확장 과제

### 3.6 Llama 3 & Llama 3.1 Distillation

- **저자:** Meta AI
- **연도:** 2024~2025
- **링크:** https://arxiv.org/abs/2407.21783 (Llama 3.1)
- **핵심 기여:**
  - 405B → 70B → 8B 단계적 증류 실전 사례
  - Logit-based + SFT 기반 하이브리드 증류
  - 8B 모델이 증류 후 이전 세대 70B 수준 도달
- **본 프로젝트에서 차용한 것:**
  - ✅ **단계적 증류(Cascaded Distillation)** 개념 인식
  - ✅ 대규모 실전 증류 사례로서 참고

### 3.7 DeepSeek-R1: Distilling Reasoning from Large Models

- **저자:** DeepSeek AI
- **연도:** 2025
- **링크:** https://arxiv.org/abs/2501.12948
- **핵심 기여:**
  - Reasoning 능력 증류에 특화
  - 671B MoE Teacher → 1.5B~70B Student 증류
  - RL + SFT + Distillation 결합
  - Qwen, Llama 등 다양한 Student 아키텍처에 적용
- **본 프로젝트에서 차용한 것:**
  - 📌 참고만: reasoning distillation은 범위 밖이지만 트렌드 인식

---

## 4. 실전 적용 & 경량화 관련

### 4.1 QLoRA: Efficient Finetuning of Quantized LLMs

- **저자:** Tim Dettmers, Artidoro Pagnoni, et al.
- **연도:** 2023
- **링크:** https://arxiv.org/abs/2305.14314
- **핵심 기여:**
  - 4-bit 양자화 + LoRA로 대형 모델 fine-tuning
  - 단일 48GB GPU에서 65B 모델 fine-tuning 가능
- **본 프로젝트에서 차용한 것:**
  - ✅ **Teacher 양자화 로딩** (8-bit/4-bit로 VRAM 절약)

### 4.2 LLM.int8(): 8-bit Matrix Multiplication for Transformers

- **저자:** Tim Dettmers, Mike Lewis, et al.
- **연도:** 2022
- **링크:** https://arxiv.org/abs/2208.07339
- **핵심 기여:**
  - 8-bit 양자화로 LLM 추론 시 메모리 절반 감소
  - bitsandbytes 라이브러리의 이론적 근거
- **본 프로젝트에서 차용한 것:**
  - ✅ **Teacher 모델 8-bit 로딩** 구현 시 참조

---

## 5. 논문 → 프로젝트 매핑

본 프로젝트의 각 설계 결정이 어느 논문에서 유래했는지 정리:

| 프로젝트 구성요소 | 차용 논문 | 차용 내용 |
|------------------|----------|----------|
| **KD Loss 수식** | Hinton et al. (2015) | `α·CE + (1-α)·T²·KL` 전체 구조 |
| **Temperature scaling** | Hinton et al. (2015) | Softmax temperature 개념 |
| **4-Way 비교 실험** | Zephyr (2023), DistilBERT (2019) | Teacher/KD/FT/Base 비교 구조 |
| **GPT-2 계열 모델 선택** | MiniLLM (2024), DistiLLM (2024) | GPT-2 계열로 벤치마크 동일 조건 |
| **Auto-regressive KD 주의점** | GKD (2023) | Train-inference mismatch 인식 |
| **Forward KL 사용** | Hinton (2015), DistiLLM (2024) | 기본 구현은 Forward KL |
| **Teacher 양자화** | QLoRA (2023), LLM.int8() (2022) | VRAM 절약을 위한 Teacher 8-bit 로드 |
| **Perplexity 평가** | DistilBERT (2019), MiniLLM (2024) | 언어모델 표준 평가 지표 |
| **데이터셋 (WikiText-2)** | MiniLLM (2024), DistiLLM (2024) | LM 증류 표준 벤치마크 |
| **White-box 증류 방식** | GKD (2023), MiniLLM (2024) | Logit 접근 가능한 오픈소스 모델 증류 |

---

## 6. 읽기 순서 추천

### 입문자

```
1. Hinton et al. (2015)          ← KD 기본 개념 (필수)
2. DistilBERT (2019)             ← NLP에서의 실전 적용
3. GKD (2023)                    ← LLM 증류의 핵심 문제
4. MiniLLM (2024)                ← LLM 특화 방법론
```

### 실전 적용 관심

```
1. Hinton et al. (2015)          ← 기본
2. Zephyr (2023)                 ← 실전 증류 파이프라인
3. Llama 3.1 (2024)              ← 대규모 실전 사례
4. QLoRA (2023)                  ← 메모리 최적화
```

### 최신 트렌드 추적

```
1. GKD (2023)                    ← On-policy distillation
2. DistiLLM (2024)               ← Skew KL divergence
3. DeepSeek-R1 (2025)            ← Reasoning distillation
4. MiniCPM (2024)                ← 소형 모델의 잠재력
```

---

## 7. 향후 추가 예정

| 상태 | 논문 / 자료 | 비고 |
|------|------------|------|
| 📋 조사 예정 | SALT (2025) — Selective Attention Layer Transfer | Layer-wise KD 최신 |
| 📋 조사 예정 | LoRA distillation 관련 논문 | LoRA + KD 결합 |
| 📋 조사 예정 | Speculative decoding + Distillation | 추론 가속 + 증류 결합 |

---

> **마지막 업데이트:** 2026-03-31  
> **관리 기준:** 프로젝트에서 직접 참조하거나 설계에 영향을 준 논문만 포함
