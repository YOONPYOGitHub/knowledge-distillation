# 코드 구조 & 파이프라인 가이드

> 프로젝트의 전체 실행 흐름과 각 모듈의 역할을 설명합니다.

---

## 1. 전체 파이프라인 흐름

```
main.py 실행
  │
  ├─ Config 로드 (YAML → KDConfig)
  │
  ├─ STEP 1: distill()         ─ KD 학습 (Teacher → Student 지식 전달)
  │
  ├─ STEP 2: train_baseline()  ─ FT 학습 (CE만으로 Student 학습, 비교 기준)
  │
  ├─ STEP 3: evaluate_all()    ─ 4-Way 평가 (PPL, 추론 속도)
  │
  └─ STEP 4: compare()         ─ 비교 차트 생성 + 자동 진단
```

### 실행 방법

```bash
# 전체 파이프라인
uv run python main.py configs/exp05_wikitext103.yaml

# 특정 단계만 실행
uv run python main.py configs/exp05_wikitext103.yaml --step distill
uv run python main.py configs/exp05_wikitext103.yaml --step evaluate --step compare
```

---

## 2. 파일 구조

```
src/
  config.py         ─ 설정 관리 (KDConfig 데이터클래스)
  dataset.py        ─ 데이터 로드 & 전처리 (Packing)
  models.py         ─ Teacher/Student 모델 로드
  distill.py        ─ Knowledge Distillation 학습
  train_baseline.py ─ Fine-tuning 학습 (비교 기준)
  evaluate.py       ─ Perplexity / 추론 속도 평가
  compare.py        ─ 비교 차트 & 자동 진단

configs/            ─ 실험별 YAML 설정 파일
results/
  checkpoints/      ─ 학습된 모델 가중치 (.pt)
  logs/             ─ 학습 로그, 평가 결과, 노트
  figures/          ─ 비교 차트 이미지
```

---

## 3. 각 모듈 상세 설명

### 3.1 config.py — 설정 관리

**역할**: 모든 하이퍼파라미터와 경로를 관리하는 중앙 설정.

**핵심 클래스: `KDConfig`**

```python
@dataclass
class KDConfig:
    # 모델
    teacher_model: str    # "gpt2-medium" 등 HuggingFace 모델명
    student_model: str    # "distilgpt2" 등

    # 데이터
    dataset_name: str     # "wikitext"
    dataset_config: str   # "wikitext-2-raw-v1" 또는 "wikitext-103-raw-v1"
    max_seq_length: int   # Packing chunking 단위 (256 등)

    # KD 하이퍼파라미터
    temperature: float    # Soft label 평탄화 정도 (높을수록 flat)
    alpha: float          # CE vs KD 비율 (0.5 = 50:50)

    # 학습
    epochs: int
    batch_size: int
    learning_rate: float

    # 경로 (run_id 기반으로 자동 생성)
    run_id: str           # 비어있으면 타임스탬프로 자동 생성
```

**디바이스 자동 감지**: CUDA > MPS > CPU 순서로 사용 가능한 디바이스를 자동 선택.

**결과 경로**: `run_id`로 실험별 디렉토리 분리.
- `results/checkpoints/{run_id}/` — 모델 가중치
- `results/logs/{run_id}/` — 로그, 평가 결과
- `results/figures/{run_id}/` — 차트

**YAML 로드**: `from_yaml("configs/exp05.yaml")` → KDConfig 객체 생성.

---

### 3.2 dataset.py — 데이터 로드 & 전처리

**역할**: WikiText 데이터셋을 로드하고, Packing 방식으로 전처리.

**주요 함수:**

| 함수 | 역할 |
|---|---|
| `load_tokenizer(model_name)` | GPT-2 계열 토크나이저 로드. pad_token이 없으면 eos_token으로 설정 |
| `load_wikitext(config, tokenizer)` | 데이터셋 로드 → Packing 전처리 |
| `create_dataloaders(config, tokenizer)` | train/validation/test DataLoader 생성 |

**Packing 방식** (논문 표준):

```
원본 텍스트:
  "Hello world." "This is a test." "Knowledge distillation is..."

1단계: 전부 토크나이즈 후 연결
  [15496, 995, 13, 1212, 318, 257, 1332, 13, 23924, 35751, ...]

2단계: max_seq_length(256) 단위로 자르기
  chunk1: [15496, 995, 13, 1212, ..., ] (256개)
  chunk2: [다음 256개]
  나머지: 버림

결과: 패딩 없음, 모든 토큰이 유효한 학습 대상
```

왜 Packing인가:
- 기존 padding 방식은 짧은 문장에 패딩이 대량 발생 → 패딩 토큰을 학습/평가하는 문제
- Packing은 패딩이 원천 차단됨. GPT-2, DistilGPT-2, MiniLLM 등 모든 LM 논문의 표준 방식

---

### 3.3 models.py — 모델 로드

**역할**: HuggingFace에서 Teacher/Student 모델을 로드.

**주요 함수:**

| 함수 | 역할 |
|---|---|
| `load_teacher(config)` | Teacher 모델 로드. `eval()` 모드 + 가중치 고정(`requires_grad=False`) |
| `load_student(config)` | Student 모델 로드. `train()` 모드 (학습 대상) |
| `model_info(model, name)` | 파라미터 수, 메모리, 디바이스 출력 |

Teacher는 학습 중 가중치가 변하지 않음 (추론만 수행). GPU에서 `fp16`이 활성화되면 반정밀도로 로드하여 메모리 절약.

---

### 3.4 distill.py — Knowledge Distillation 학습

**역할**: Teacher의 지식을 Student에게 전달하는 핵심 학습 모듈.

**KD Loss 수식:**

$$L_{total} = \alpha \cdot L_{CE} + (1 - \alpha) \cdot T^2 \cdot L_{KD}$$

- $L_{CE}$: Student 예측 vs **정답** (Hard Label)
- $L_{KD}$: Student 분포 vs **Teacher 분포** (Soft Label)
- $T$: Temperature — 높을수록 분포가 평탄해져 dark knowledge 전달
- $\alpha$: CE와 KD의 비율

**주요 함수:**

| 함수 | 역할 |
|---|---|
| `kd_loss(student_logits, teacher_logits, labels, config)` | KD Loss 계산. Label shift 적용 |
| `train_one_epoch(teacher, student, dataloader, optimizer, config)` | 1 epoch 학습 루프 |
| `validate(teacher, student, dataloader, config)` | 검증 데이터로 loss 계산 (학습 없음) |
| `distill(config)` | 전체 KD 파이프라인 (데이터로드 → 학습 → 저장) |

**Label Shift** (Causal LM 필수):

```
입력:    [A, B, C, D, E]
logits:  [?, ?, ?, ?, ?]  ← 각 위치에서의 예측

정상: logits[0] → labels[1] (A 다음에 B가 오는지 예측)
     logits[1] → labels[2] (B 다음에 C가 오는지 예측)

코드:
  shift_logits = logits[..., :-1, :]   # [A, B, C, D]의 예측
  shift_labels = labels[..., 1:]       # [B, C, D, E] 정답
```

**학습 루프 흐름:**

```
매 배치(batch):
  1. Teacher forward (no_grad)  → Teacher 출력 분포 (고정)
  2. Student forward            → Student 출력 분포 (학습 대상)
  3. Label shift 적용
  4. CE Loss 계산: Student예측 vs 정답
  5. KD Loss 계산: KL(Student분포 ∥ Teacher분포) × T²
  6. Total Loss = α·CE + (1-α)·KD
  7. 역전파(backprop) → Student 가중치 업데이트

매 epoch 끝:
  - Validation loss 계산
  - Val loss가 이전 최고보다 낮으면 → student_kd_best.pt 저장 (덮어쓰기)
```

**산출물:**
- `student_kd_best.pt` — 가장 좋은 epoch의 Student 가중치
- `distill_history.json` — epoch별 loss 기록

---

### 3.5 train_baseline.py — Fine-tuning 학습

**역할**: KD 없이 CE Loss만으로 Student를 학습. KD 효과의 **비교 기준**(baseline).

distill.py와의 차이:

| | distill.py (KD) | train_baseline.py (FT) |
|---|---|---|
| Teacher 사용 | ✅ 매 배치 forward | ❌ 없음 |
| Loss | α·CE + (1-α)·KD | CE만 |
| 학습 속도 | 느림 (Teacher forward 오버헤드) | 빠름 |
| 산출물 | student_kd_best.pt | student_ft_best.pt |

동일한 데이터, 동일한 Student, 동일한 optimizer로 학습하므로 **Loss 함수만 다른 공정 비교**.

---

### 3.6 evaluate.py — 모델 평가

**역할**: 4개 모델의 Perplexity(PPL)와 추론 속도를 측정.

**평가 대상 4-Way:**

| # | 모델 | 설명 |
|---|---|---|
| 1 | Teacher | 원본 대형 모델 (gpt2-medium 등) |
| 2 | Student (KD) | KD 학습된 Student (student_kd_best.pt) |
| 3 | Student (FT) | FT 학습된 Student (student_ft_best.pt) |
| 4 | Student (Base) | 추가 학습 없는 Student 원본 |

**주요 함수:**

| 함수 | 역할 |
|---|---|
| `evaluate_perplexity(model, dataloader, device)` | PPL 계산: $PPL = e^{avg\_CE\_loss}$. Label shift 적용 |
| `evaluate_speed(model, dataloader, device)` | 추론 속도 측정: tokens/sec, ms/token |
| `evaluate_model(model, name, dataloader, device)` | 단일 모델 전체 평가 (PPL + 속도 + 크기) |
| `evaluate_all(config)` | 4-Way 전체 평가 실행 |

**PPL(Perplexity)의 의미:**
- "모델이 다음 토큰을 고를 때 평균 몇 개 후보 중에서 헷갈리는가"
- PPL=30이면 50,257개 vocab 중 30개로 좁힘 → 좋은 성능
- 낮을수록 좋음

**산출물:** `evaluation_results.json`

---

### 3.7 compare.py — 비교 & 시각화

**역할**: 평가 결과를 비교 테이블, 차트, 자동 진단으로 정리.

**주요 함수:**

| 함수 | 역할 |
|---|---|
| `print_comparison_table(results)` | 4-Way 비교 테이블 콘솔 출력 |
| `plot_perplexity(results, save_path)` | PPL 비교 막대 그래프 |
| `plot_speed(results, save_path)` | 추론 속도 비교 막대 그래프 |
| `plot_training_curves(config, save_path)` | KD vs FT 학습 Loss 곡선 |
| `generate_summary(results, config)` | 자동 진단 포함 summary.json 생성 |
| `compare(config)` | 전체 비교 실행 |

**자동 진단 규칙:**
- PPL > 10,000 → "비정상적으로 높음 (패딩 과다 또는 미학습)"
- PPL < 1.5 → "과적합 의심 (패딩 토큰 예측으로 loss 왜곡 가능)"
- KD PPL >= FT PPL → "증류 효과 미확인"
- KD PPL < FT PPL → "증류 효과 확인됨"

**산출물:**
- `perplexity_comparison.png` — PPL 비교 차트
- `inference_speed.png` — 속도 비교 차트
- `training_loss.png` — 학습 곡선 차트
- `summary.json` — 설정 + 결과 + 진단 + 다음 액션
- `notes.md` — 실험 노트 템플릿 (첫 생성 시에만)

---

## 4. 산출물 구조

하나의 실험을 수행하면 아래 파일이 생성됩니다:

```
results/
  checkpoints/{run_id}/
    student_kd_best.pt          ← KD 최적 가중치
    student_ft_best.pt          ← FT 최적 가중치

  logs/{run_id}/
    distill_history.json        ← KD epoch별: total_loss, ce_loss, kd_loss, val_ce_loss, time
    baseline_history.json       ← FT epoch별: ce_loss, val_ce_loss, time
    evaluation_results.json     ← 4-Way: PPL, tokens/sec, 파라미터 수
    summary.json                ← config + 결과 + 자동 진단 + 다음 액션
    notes.md                    ← 실험 노트 (자동 생성 + 수동 메모)

  figures/{run_id}/
    perplexity_comparison.png   ← PPL 비교 막대 그래프
    inference_speed.png         ← 속도 비교 막대 그래프
    training_loss.png           ← KD vs FT 학습 곡선
```

---

## 5. 핵심 개념 요약

### Knowledge Distillation이란

큰 모델(Teacher)의 지식을 작은 모델(Student)에게 전달하는 기법.

```
Teacher (gpt2-medium, 355M)
  │
  │  "다음 토큰은 A가 70%, B가 20%, C가 10% 확률"  ← Soft Label (dark knowledge)
  │
  ▼
Student (distilgpt2, 82M)
  │
  │  이 분포를 모방하도록 학습
  │  → 정답(A)뿐 아니라 "B, C도 가능"이라는 정보까지 학습
  │
  결과: 작지만 Teacher의 판단력을 물려받은 모델
```

### FT(Fine-tuning)와의 차이

```
FT:  정답만 학습    → "A가 정답" (O/X만 알려줌)
KD:  분포를 학습    → "A가 70%, B가 20%, C가 10%" (뉘앙스까지)
```

FT는 빠르고 단순하지만, 데이터가 크면 외울 수 없어서 일반화에 약함.
KD는 Teacher의 분포 정보로 일반화 능력을 확보 → 큰 데이터에서 유리.

### Temperature (T)의 역할

```
T=1 (sharp):  [0.90, 0.05, 0.03, 0.02, ...]  ← 거의 정답만 높음
T=4 (soft):   [0.40, 0.25, 0.20, 0.15, ...]  ← 후보들의 차이가 완화됨
```

T를 높이면 Teacher의 출력이 평탄해져서, "정답은 아니지만 유력한 후보"(dark knowledge)가 드러남.
단, 너무 높으면 노이즈에 가까워짐.

### Alpha (α)의 역할

$$L = \alpha \cdot CE + (1 - \alpha) \cdot KD$$

- α=1.0: FT와 동일 (CE만)
- α=0.5: CE 50% + KD 50% (균형)
- α=0.0: KD만 (정답 무시, Teacher만 모방)

---

## 6. 용어 설명

이 문서에서 사용하는 용어를 간단히 정리합니다.

### 데이터 전처리

| 용어 | 설명 |
|---|---|
| **Padding (패딩)** | 길이가 다른 텍스트들을 같은 길이로 맞추기 위해 **빈칸(pad token)을 채워넣는** 방식. 짧은 문장에 패딩이 대량 발생하면 모델이 "빈칸 예측"을 학습하게 되어 PPL이 왜곡됩니다. |
| **Packing (패킹)** | 모든 텍스트를 하나로 이어붙인 후 **고정 길이로 잘라내는** 방식. 패딩이 전혀 없으므로 모든 토큰이 유의미한 학습 대상이 됩니다. 이 프로젝트에서 사용하는 방식. |
| **Tokenize (토크나이즈)** | 텍스트를 모델이 이해할 수 있는 **숫자(토큰 ID)로 변환**하는 과정. "Hello world" → `[15496, 995]` |

```
패딩 방식 (문제 있음):              패킹 방식 (이 프로젝트):
"Hello world"  → [15496, 995, PAD, PAD]    모든 텍스트를 하나로 연결:
"Hi"           → [17250, PAD, PAD, PAD]    [15496, 995, 17250, 8496, ...]
"Good morning" → [11274, 8496, PAD, PAD]   → 256개씩 자르기 (나머지 버림)
                  ↑ PAD가 학습을 오염시킴      ↑ 모든 토큰이 유효
```

### 학습 파라미터

| 용어 | 설명 | 이 프로젝트 기본값 |
|---|---|---|
| **Epoch (에폭)** | 전체 학습 데이터를 **1번 처음부터 끝까지 돌리는 것**. 3 epochs = 전체 데이터 3바퀴. 너무 적으면 덜 배우고, 너무 많으면 외워버림(과적합). | 3~10 |
| **Batch Size (배치 크기)** | 한 번에 모델에 넣는 **데이터 묶음 크기**. 전체 데이터를 한 번에 넣을 수 없어서 작은 묶음으로 나눠 학습. 클수록 안정적이지만 메모리를 많이 씀. | 8 |
| **Learning Rate (학습률)** | 모델 가중치를 **한 번에 얼마나 크게 바꿀지** 결정하는 값. 너무 크면 최적점을 지나치고, 너무 작으면 학습이 느림. | 5e-5 (=0.00005) |
| **max_seq_length** | Packing 시 텍스트를 자르는 **고정 길이**. 이 길이 단위로 chunk를 만듦. | 256 토큰 |

```
데이터 전체: 10,000개 샘플
batch_size=8이면 → 1,250번 업데이트해야 1 epoch
epochs=3이면   → 총 3,750번 업데이트

learning_rate=5e-5 → 매 업데이트마다 가중치를 0.00005만큼 조정
(큰 것 같지만, 수십억 개의 가중치가 각각 미세하게 변하는 것)
```

### 모델 관련

| 용어 | 설명 |
|---|---|
| **Parameter (파라미터)** | 모델 내부의 **학습 가능한 숫자(가중치)**. "82M 파라미터" = 약 8,200만 개의 숫자가 모델 안에 있다는 뜻. 많을수록 똑똑하지만 크고 느림. |
| **Checkpoint (체크포인트)** | 학습 중간에 모델 가중치를 **파일로 저장**한 것 (`.pt` 파일). 가장 좋았던 시점의 모델을 나중에 불러올 수 있음. |
| **Forward** | 입력 데이터를 모델에 넣어서 **예측 결과를 얻는** 과정. Teacher는 forward만 하고 (추론), Student는 forward → loss → backward 순서로 학습. |
| **Backward (역전파)** | Loss(오차)를 기반으로 **가중치를 어떻게 바꿔야 하는지 계산**하는 과정. 이 정보로 optimizer가 가중치를 실제 업데이트. |
| **Optimizer (옵티마이저)** | 계산된 gradient를 이용해 **실제로 가중치를 업데이트하는 알고리즘**. 이 프로젝트는 AdamW 사용. |

### 평가 관련

| 용어 | 설명 |
|---|---|
| **PPL (Perplexity)** | 모델 품질 지표. "다음 단어를 고를 때 평균 몇 개 후보 중 고민하나". **낮을수록 좋음**. PPL=30이면 50,257개 단어 중 30개로 좁힌 것. |
| **Validation (검증)** | 학습에 사용하지 않은 별도 데이터로 **모델 성능을 중간 점검**하는 것. 과적합 여부를 판단하는 기준. |
| **Label Shift** | Causal LM에서 logits과 labels를 **한 칸 밀어서** 맞추는 필수 처리. 위치 $n$의 예측 → 위치 $n+1$의 정답과 비교. |
| **tokens/sec** | 모델이 1초에 처리하는 토큰 수. **높을수록 빠름**. 작은 모델이 큰 모델보다 빠름. |

### Loss 관련

| 용어 | 설명 |
|---|---|
| **Loss (손실)** | 모델 예측이 정답에서 **얼마나 벗어났는지** 나타내는 숫자. 학습의 목표는 이것을 줄이는 것. |
| **CE Loss (Cross-Entropy)** | 모델 예측 vs **정답** 사이의 차이. 일반 학습(FT)에서 사용하는 기본 Loss. |
| **KD Loss** | 모델 예측 vs **Teacher 분포** 사이의 차이(KL Divergence). Student가 Teacher를 따라하게 만드는 Loss. |
| **KL Divergence** | 두 확률 분포가 **얼마나 다른지** 측정. 0이면 완전 동일. KD에서는 Student와 Teacher 분포의 차이를 측정. |

### 기타

| 용어 | 설명 |
|---|---|
| **DataLoader** | 데이터를 batch 단위로 잘라서 모델에 **순서대로 공급**하는 도구. |
| **YAML** | 사람이 읽기 쉬운 **설정 파일 형식**. `configs/` 폴더의 실험 설정에 사용. |
| **run_id** | 각 실험을 구분하는 **고유 ID**. 타임스탬프 기반 (예: `20260404_082410`). 결과 폴더명에 사용. |
| **eval() 모드** | 모델을 **추론 전용**으로 설정. Dropout 비활성화 등. Teacher가 이 모드로 동작. |
| **no_grad** | 역전파 계산을 **끄는** 설정. Teacher forward 시 사용. 메모리 절약 + 속도 향상. |

> 더 상세한 용어 설명은 [glossary.md](glossary.md) 에 정리되어 있습니다.
