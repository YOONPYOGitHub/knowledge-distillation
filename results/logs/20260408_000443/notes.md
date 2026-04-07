# 실험 노트 — 20260408_000443 (exp06: Teacher FT + KD, WikiText-2)

## 실험 설정

| 항목 | 값 |
|------|-----|
| Teacher | gpt2 (124M) → **WikiText-2에 fine-tune 후 KD** |
| Student | distilgpt2 (82M) |
| Dataset | WikiText-2 (~2M tokens) |
| Temperature | 2.0 |
| Alpha | 0.5 |
| Epochs | Teacher FT 3ep / Student KD·FT 3ep |
| Batch Size | 4 |
| Max Seq Length | 256 |
| Device | MPS (Apple Silicon) |

## 결과 요약

| 모델 | Perplexity (↓) | Tokens/sec (↑) | Size |
|------|---------------|----------------|------|
| **Teacher (FT'd gpt2)** | **25.07** | 14,113 | 498 MB |
| Student (KD) | 33.00 | 22,209 | 328 MB |
| **Student (FT)** | **32.56** | 22,172 | 328 MB |
| Student (Base) | 65.27 | 22,021 | 328 MB |

## 학습 이력

### Teacher FT (에폭당 ~11.3분, 총 ~34분)

| Epoch | Train CE | Val CE | Time |
|-------|----------|--------|------|
| 1 | 3.414 | 3.257 | 680s |
| 2 | 3.220 | 3.242 | 679s |
| 3 | 3.117 | 3.237 | 673s |

### KD (에폭당 ~12분, 총 ~36분)

| Epoch | Train CE | Val CE | KD Loss | Time |
|-------|----------|--------|---------|------|
| 1 | 3.608 | 3.545 | 183.49 | 719s |
| 2 | 3.464 | 3.520 | 149.39 | 718s |
| 3 | 3.408 | 3.506 | 138.74 | 717s |

### SFT (에폭당 ~7.4분, 총 ~22분)

| Epoch | Train CE | Val CE | Time |
|-------|----------|--------|------|
| 1 | 3.644 | 3.508 | 445s |
| 2 | 3.397 | 3.496 | 445s |
| 3 | 3.257 | 3.506 | 444s |

## Teacher FT 효과 — exp04(Teacher FT 없음) vs exp06(Teacher FT 있음) 비교

| 모델 | exp04 (FT ❌) | exp06 (FT ✅) | 변화 |
|------|-------------|-------------|------|
| Teacher | 43.60 | **25.07** | **-42.5%** |
| Student (KD) | 48.71 | **33.00** | **-32.3%** |
| Student (FT) | 32.69 | 32.56 | -0.4% |
| KD vs FT 격차 | **16.02** | **0.44** | **격차 97% 감소** |

## 핵심 분석

1. **Teacher FT가 KD의 핵심 전제임을 강력히 확인**: Teacher FT만으로 KD-FT 격차가 16.02 → 0.44로 97% 감소
2. **KD(33.00) ≈ FT(32.56)**: PPL 차이 0.44는 거의 동등 수준. 아직 KD가 소폭 뒤지지만 역전 가능 범위
3. **Teacher PPL(25.07) < Student FT PPL(32.56)**: Teacher가 student보다 확실히 강해야 한다는 조건 충족
4. **KD val CE가 에폭 3에서 아직 하락 중** (3.545 → 3.520 → 3.506): 에폭 늘리면 추가 개선 가능
5. **SFT val CE는 에폭 2에서 이미 정체** (3.508 → 3.496 → 3.506): 오버피팅 시작

## exp07 방향

- α를 0.5→0.3으로 낮춰 teacher soft target 비중 확대 (teacher가 이제 강하므로)
- KD 에폭을 5로 늘려 수렴 확인 (KD는 아직 개선 중, SFT는 이미 정체)
- gpt2-medium teacher + WikiText-103으로 스케일업

