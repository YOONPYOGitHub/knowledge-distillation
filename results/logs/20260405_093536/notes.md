# 실험 노트 — 20260405_093536

## 자동 진단

- ⚠️ KD(47.31) >= FT(32.77) — 증류 효과 미확인

## 다음 액션

- [ ] temperature 값 조정 (현재 → +2 시도)
- [ ] alpha 값 낮추기 (CE 비율 줄이기)
- [ ] GPU 서버에서 server_config()로 본 실험 수행

## 메모

### 1. 실험 조건 요약

| 항목 | 값 | exp03 대비 변경 |
|---|---|---|
| **Teacher** | **gpt2-medium (355M, 1.4GB)** | gpt2(124M) → **gpt2-medium(355M)** |
| Student | distilgpt2 (82M, 327.7MB) | - |
| Teacher/Student 비율 | **4.3배** | 1.5배 → **4.3배** |
| Temperature | 2.0 | - |
| Alpha (α) | 0.5 | - |
| Epochs | 3 | - |
| Batch Size | 4 | - |
| max_seq_length | 256 | - |
| Device | Apple MPS (MacBook) | - |

### 2. 4-Way 비교 결과

| 모델 | exp03 PPL (gpt2) | **exp04 PPL (gpt2-medium)** | 변화 |
|---|---|---|---|
| **Teacher** | 43.60 (gpt2) | **31.32** (gpt2-medium) | **-12.28** (대폭 개선) |
| **Student (KD)** | 48.71 | **47.31** | -1.40 (소폭 개선) |
| Student (FT) | 32.69 | 32.77 | +0.08 (동일) |
| Student (Base) | 65.27 | 65.27 | 동일 |

### 3. KD 학습 수렴 추이

| Epoch | CE Loss | KD Loss | Val CE | 시간 |
|---|---|---|---|---|
| 1 | 4.06 | 243.40 | 3.90 | ~17분 |
| 2 | 3.97 | 219.39 | 3.88 | ~17분 |
| 3 | 3.93 | 207.51 | 3.87 | ~17분 |

- Teacher가 커진 만큼 KD loss가 대폭 증가 (exp03: 103 → exp04: 207)
- Teacher-Student 분포 차이가 2배로 벌어짐 → Student가 모방해야 할 양이 늘어남
- Val CE는 3.90→3.87로 꾸준히 감소 → 수렴하지 않았을 가능성
- epoch당 17분 (exp03 대비 +5분) — gpt2-medium forward pass 오버헤드

### 4. FT 학습 결과

| Epoch | Train CE | Val CE | 시간 |
|---|---|---|---|
| 1 | 3.64 | 3.51 | ~7분 |
| 2 | 3.40 | 3.50 | ~7분 |
| 3 | 3.26 | 3.50 | ~7분 |

- exp03과 동일한 패턴 (Teacher 변경이 FT에 영향 없음 — FT는 Teacher를 사용하지 않으므로 당연)
- Epoch 2부터 Val CE 정체 → 과적합 경계

### 5. 핵심 발견 — Teacher가 좋아져도 KD 개선은 미미

| 실험 | Teacher | Teacher PPL | KD PPL | FT PPL | KD-FT 갭 |
|---|---|---|---|---|---|
| exp02 | gpt2 (124M) | 43.60 | 48.96 | 32.56 | +16.40 |
| exp03 | gpt2 (124M) | 43.60 | 48.71 | 32.69 | +16.02 |
| **exp04** | **gpt2-medium (355M)** | **31.32** | **47.31** | 32.77 | **+14.54** |

- Teacher PPL이 43.6→31.3으로 **12.3 개선**됐지만, KD PPL은 48.7→47.3으로 **1.4만 개선**
- Teacher 개선분의 **11%만 Student에게 전달**됨
- KD-FT 갭은 16.0→14.5로 줄어드는 추세이긴 함

### 6. 원인 분석 — 왜 Teacher 개선이 Student에 전달되지 않는가

**① Teacher-Student 아키텍처 갭**
- gpt2-medium: 24층, hidden=1024, head=16
- distilgpt2: 6층, hidden=768, head=12
- Teacher의 깊은 표현(24층 거쳐 만든 분포)을 Student(6층)가 모방하는 데 구조적 한계
- 층 수 4배, hidden 차이 1.3배 → Student가 수용할 수 있는 정보의 상한이 존재

**② WikiText-2의 데이터 크기 한계 (여전히)**
- 2M 토큰에서 FT가 외우기 가능 → PPL=32.8
- KD는 Teacher 모방이라는 추가 목표 때문에 같은 데이터에서 CE 수렴이 느림
- 더 큰 데이터에서는 FT의 외우기가 불가능해지고 KD의 일반화 효과가 부각될 수 있음

**③ KD가 아직 수렴하지 않았을 가능성**
- KD loss: 243→219→207 — 여전히 빠르게 감소 중
- exp03에서 KD loss가 103까지 내려갔는데 이번엔 207 → epoch이 더 필요할 수 있음
- Teacher가 커질수록 수렴에 필요한 epoch도 늘어남

### 7. 전체 실험 흐름 정리

| 실험 | 변경점 | KD PPL | 주요 발견 |
|---|---|---|---|
| exp01 | 코드 수정 전 (패딩+shift 없음) | 128.22 | PPL 수치 자체가 무의미 |
| exp02 | **Packing + Label Shift 적용** | 48.96 | PPL 정상화 (논문 수준) |
| exp03 | T=4→2, α=0.3→0.5 | 48.71 | T/α 튜닝 효과 없음 |
| **exp04** | **Teacher: gpt2→gpt2-medium** | **47.31** | Teacher 개선 효과 미미 (11%만 전달) |

### 8. 다음 실험 방향

T/α 튜닝(exp03)과 Teacher 스케일업(exp04) 모두 효과가 제한적이었으므로, 남은 변수는 **데이터 크기**.

#### 가설: WikiText-103에서 KD의 일반화 효과가 드러남

- WikiText-2(현재): FT가 데이터를 외울 수 있어서 KD 대비 유리
- WikiText-103(50배): 외우기 불가 → FT PPL 상승 + KD 일반화 능력으로 갭 역전 기대
- 추가: gpt2-medium Teacher + WikiText-103 조합이면 두 효과 동시 검증

#### 맥북에서 가능한 대안

- WikiText-103은 epoch당 ~10시간 (맥북 비현실적, GPU 서버 필요)
- 맥북에서 가능한 시도: epochs를 5~7로 늘려 KD 수렴 확인 (KD loss가 아직 감소 중이므로)

| 방향 | 기대 | 소요 | 환경 |
|---|---|---|---|
| exp04 config + epochs=7 | KD 수렴 확인, PPL 추가 하락 | ~4시간 | 맥북 |
| WikiText-103 + gpt2-medium | KD > FT 역전 가능성 | ~30시간+ | GPU 서버 |

