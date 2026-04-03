# 모델 선택 가이드

> Teacher / Student 모델 조합 선택을 위한 상세 가이드

## 1. 모델 선택 기준

지식 증류에서 모델을 선택할 때 고려해야 할 핵심 요소:

| 기준 | 설명 |
|------|------|
| **아키텍처 호환성** | Teacher와 Student가 동일 계열(GPT-2 ↔ GPT-2)이면 증류 효과가 높음 |
| **크기 비율** | Teacher:Student 파라미터 비율이 3~10x 정도가 적당 |
| **GPU VRAM** | Teacher(추론) + Student(학습) 동시 로딩 가능해야 함 |
| **토크나이저 호환** | 동일 토크나이저를 사용하면 구현이 단순해짐 |
| **라이선스** | 상용/연구용 라이선스 확인 필수 |

## 2. 추천 모델 조합

### Tier 1: 입문용 (단일 GPU 12GB~24GB)

> **GPT-2 계열** — 동일 토크나이저, 동일 아키텍처, 가벼움

| 조합 | Teacher | Student | 비율 | 동시 VRAM (FP16) | 난이도 |
|------|---------|---------|------|-----------------|--------|
| **A (최소)** | `gpt2-medium` (345M) | `distilgpt2` (82M) | 4.2x | ~2.0GB | ⭐ |
| **B (권장)** | `gpt2-large` (774M) | `gpt2` (124M) | 6.2x | ~3.5GB | ⭐⭐ |
| **C (강화)** | `gpt2-xl` (1.5B) | `gpt2-medium` (345M) | 4.3x | ~7.0GB | ⭐⭐ |

**장점:**
- 동일 토크나이저 (`gpt2` tokenizer) → 별도 토큰 매핑 불필요
- 동일 아키텍처 (Transformer Decoder) → logit 차원이 동일
- 모든 모델이 Hugging Face에서 무제한 다운로드 가능
- 가벼워서 일반 GPU에서도 실험 가능

**단점:**
- 절대적인 모델 성능이 최신 LLM 대비 낮음
- 한국어 성능이 약함 (영어 중심으로 학습됨)

### Tier 2: 중급 (24GB~48GB GPU)

> **LLaMA / Mistral 계열** — 최신 고성능 오픈소스 모델

| 조합 | Teacher | Student | 비율 | 동시 VRAM (FP16) | 난이도 |
|------|---------|---------|------|-----------------|--------|
| **D** | `mistralai/Mistral-7B-v0.1` (7B) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B) | 6.4x | ~16GB | ⭐⭐⭐ |
| **E** | `meta-llama/Llama-2-7b-hf` (7B) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B) | 6.4x | ~16GB | ⭐⭐⭐ |
| **F** | `meta-llama/Llama-2-13b-hf` (13B) | `meta-llama/Llama-2-7b-hf` (7B) | 1.9x | ~40GB | ⭐⭐⭐⭐ |

**장점:**
- 최신 모델로 절대 성능이 높음
- 실제 서비스 적용에 가까운 실험 가능
- 커뮤니티 활발, 참고 자료 풍부

**단점:**
- 높은 GPU 메모리 요구
- LLaMA 계열은 Meta 이용약관 동의 필요 (HF 토큰 인증)
- 학습 시간이 Tier 1 대비 크게 증가

### Tier 3: 고급 (Multi-GPU / A100 80GB)

| 조합 | Teacher | Student | 비율 | 동시 VRAM (FP16) | 난이도 |
|------|---------|---------|------|-----------------|--------|
| **G** | `meta-llama/Llama-2-70b-hf` (70B) | `meta-llama/Llama-2-7b-hf` (7B) | 10x | ~160GB | ⭐⭐⭐⭐⭐ |
| **H** | `Qwen/Qwen2-72B` (72B) | `Qwen/Qwen2-7B` (7B) | 10x | ~160GB | ⭐⭐⭐⭐⭐ |

> ⚠️ Tier 3는 다중 GPU 환경, DeepSpeed/FSDP 분산 학습이 필수

## 3. 모델별 상세 스펙

### GPT-2 계열

| 모델 | 파라미터 | Layers | Heads | Hidden | Vocab | Context |
|------|---------|--------|-------|--------|-------|---------|
| `distilgpt2` | 82M | 6 | 12 | 768 | 50,257 | 1024 |
| `gpt2` | 124M | 12 | 12 | 768 | 50,257 | 1024 |
| `gpt2-medium` | 345M | 24 | 16 | 1024 | 50,257 | 1024 |
| `gpt2-large` | 774M | 36 | 20 | 1280 | 50,257 | 1024 |
| `gpt2-xl` | 1.5B | 48 | 25 | 1600 | 50,257 | 1024 |

### LLaMA / Mistral 계열

| 모델 | 파라미터 | Layers | Heads | Hidden | Vocab | Context |
|------|---------|--------|-------|--------|-------|---------|
| `TinyLlama-1.1B` | 1.1B | 22 | 32 | 2048 | 32,000 | 2048 |
| `Llama-2-7b` | 6.7B | 32 | 32 | 4096 | 32,000 | 4096 |
| `Llama-2-13b` | 13B | 40 | 40 | 5120 | 32,000 | 4096 |
| `Mistral-7B-v0.1` | 7.2B | 32 | 32 | 4096 | 32,000 | 8192 |

## 4. 토크나이저 호환성 매트릭스

지식 증류에서는 **Teacher와 Student의 토크나이저가 동일**해야 logit 차원이 맞습니다.

| Teacher ↓ / Student → | distilgpt2 | gpt2 | gpt2-medium | TinyLlama | Llama-2-7b |
|------------------------|:----------:|:----:|:-----------:|:---------:|:----------:|
| gpt2-medium | ✅ | ✅ | - | ❌ | ❌ |
| gpt2-large | ✅ | ✅ | ✅ | ❌ | ❌ |
| gpt2-xl | ✅ | ✅ | ✅ | ❌ | ❌ |
| Mistral-7B | ❌ | ❌ | ❌ | ⚠️ | ❌ |
| Llama-2-7b | ❌ | ❌ | ❌ | ✅ | - |
| Llama-2-13b | ❌ | ❌ | ❌ | ✅ | ✅ |

- ✅ 동일 토크나이저, 바로 사용 가능
- ⚠️ 유사하지만 vocab 크기 차이 있을 수 있음 (projection layer 필요)
- ❌ 다른 토크나이저, 별도 매핑 로직 필요

## 5. GPU 환경별 권장 조합

### 내 GPU 확인 방법

```bash
# NVIDIA GPU 확인 (GPU 서버)
nvidia-smi

# VRAM 용량 확인
nvidia-smi --query-gpu=name,memory.total --format=csv

# Apple Silicon MPS 확인 (맥북)
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

### 환경별 추천

| 환경 | GPU / 칩 | 추천 조합 | 비고 |
|------|-----------|----------|------|
| **맥북 (Apple M4, 16GB)** | MPS | 조합 A (`gpt2-medium` → `distilgpt2`) | 디버깅/테스트용, batch_size 작게 |
| **맥북 (Apple M4 Pro+, 24GB+)** | MPS | 조합 A 또는 B | 소규모 실험 가능 |
| RTX 4060 | 8GB | 조합 A (`gpt2-medium` → `distilgpt2`) | FP16 필수 |
| RTX 4070 | 12GB | 조합 A 또는 B | FP16 권장 |
| RTX 4080 | 16GB | 조합 B (`gpt2-large` → `gpt2`) | 여유 있음 |
| RTX 4090 | 24GB | 조합 B 또는 C  | 속도 빠름 |
| RTX 5090 | 32GB | 조합 C 또는 D | 최신 소비자 GPU |
| A100 | 40GB | 조합 D (`Mistral-7B` → `TinyLlama`) | 전문 실험 |
| A100 | 80GB | 조합 E, F | 대규모 실험 |
| H100 | 80GB | 조합 F | 최대 단일 GPU |
| Multi-GPU (2x A100/H100) | 160GB+ | 조합 G, H | Tier 3 분산 학습 필수 |

> 팁: 맥북에서 코드 디버깅 → GPU 서버에서 본 학습 흐름이면, 맥북에서는 조합 A로 빠르게 테스트하고 서버에서 조합 B~D로 본 실험하세요.

### VRAM이 부족할 때 대처법

1. **FP16 / BF16 사용** — VRAM 사용량 절반으로 감소
2. **Gradient Checkpointing** — VRAM 절약 (속도는 약간 느려짐)
3. **Batch Size 축소** — 가장 간단한 방법
4. **Teacher를 8bit 양자화** — `bitsandbytes` 라이브러리로 추론 시 VRAM 크게 절약
5. **Gradient Accumulation** — 작은 배치를 누적하여 큰 배치 효과

```python
# 8-bit 양자화로 Teacher 로드 (VRAM 절약)
from transformers import AutoModelForCausalLM

teacher = AutoModelForCausalLM.from_pretrained(
    "gpt2-large",
    load_in_8bit=True,
    device_map="auto"
)
```

## 6. 이 프로젝트의 기본 선택

본 프로젝트는 **조합 B**를 기본으로 사용합니다:

| 역할 | 모델 | 이유 |
|------|------|------|
| **Teacher** | `gpt2-large` (774M) | 충분한 크기 차이, 24GB GPU에서 여유, 무제한 다운로드 |
| **Student** | `gpt2` (124M) | 동일 토크나이저, 6.2x 크기비, 빠른 학습 |

> 환경에 따라 [모델 설치 가이드](./model-installation-guide.md)를 참고하여 다른 조합으로 변경할 수 있습니다.
