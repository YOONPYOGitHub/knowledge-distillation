# 실험 노트 — 20260404_082410

## 자동 진단

- ⚠️ KD(48.96) >= FT(32.56) — 증류 효과 미확인

## 다음 액션

- [ ] temperature 값 조정 (현재 → +2 시도)
- [ ] alpha 값 낮추기 (CE 비율 줄이기)
- [ ] GPU 서버에서 server_config()로 본 실험 수행

## 메모

### 1. 실험 조건 요약

| 항목 | 값 | 비고 |
|---|---|---|
| Teacher | gpt2 (124M, 497.8MB) | |
| Student | distilgpt2 (82M, 327.7MB) | |
| 압축률 | 파라미터 34% 감소, 용량 34% 감소 | |
| Temperature | 4.0 | |
| Alpha (α) | 0.3 → KD Loss = 0.3·CE + 0.7·T²·KL | |
| Epochs | 3 | |
| Batch Size | 4 | |
| max_seq_length | 256 | |
| Device | Apple MPS (MacBook) | |
| **코드 변경** | Packing + Label Shift 적용 | exp02 대비 핵심 변경 |

### 2. 코드 수정 효과 — exp02 vs exp03

| 모델 | exp02 PPL (수정 전) | exp03 PPL (수정 후) | 변화 |
|---|---|---|---|
| Teacher (GPT-2) | 32,574 | **43.60** | 논문 기준(~29) 근처로 정상화 |
| Student (KD) | 128.22 | **48.96** | 의미 있는 절대 수치 |
| Student (FT) | 1.00 | **32.56** | 패딩 암기 → 실제 언어 모델링 |
| Student (Base) | 36,266 | **65.27** | distilgpt2 사전학습 수준 |

> **핵심**: Packing + Label Shift 적용으로 PPL이 논문 수준으로 정상화됨.
> exp02까지의 PPL은 패딩 토큰 학습 + shift 누락으로 의미 없는 수치였음.
> 이제 타 논문과 절대적 비교가 가능한 상태.

### 3. 4-Way 비교 결과

| 모델 | PPL | tokens/sec | Teacher 대비 속도 |
|---|---|---|---|
| Teacher (GPT-2) | 43.60 | 14,146 | 1.0x |
| Student (KD) | 48.96 | 22,288 | **1.58x** |
| Student (FT) | **32.56** | 22,070 | **1.56x** |
| Student (Base) | 65.27 | 22,295 | 1.58x |

**모델별 해석:**
- **Teacher PPL=43.60**: GPT-2 논문(29.41)보다 높은 이유는 seq=256 packing에서의 평가 조건 차이. 합리적 범위
- **Student (KD) PPL=48.96**: Base(65.27) 대비 25% 개선. Teacher(43.60)에 근접하지만 아직 초과
- **Student (FT) PPL=32.56**: Base 대비 50% 개선. Teacher보다도 낮음 — CE-only 학습이 이 조건에서 더 효과적
- **Student (Base) PPL=65.27**: 추가 학습 없는 distilgpt2 원본 성능 기준선

### 4. KD 학습 수렴 추이

| Epoch | CE Loss | KD Loss | Val CE | 시간 |
|---|---|---|---|---|
| 1 | 4.11 | 113.14 | 3.94 | ~12분 |
| 2 | 4.04 | 103.73 | 3.92 | ~12분 |
| 3 | 4.02 | 99.13 | 3.91 | ~12분 |

- KD Loss 감소: 113→99 (-12.4%) — Teacher 모방 개선 중이나 여전히 높음
- Val CE 감소폭 축소: 3.94→3.92→3.91 — 수렴에 가까움
- 학습 시간: **exp02 대비 60% 단축** (30분→12분/epoch) — Packing으로 패딩 제거 → 유효 데이터 비율 증가

### 5. FT 학습 결과

| Epoch | Train CE | Val CE | 시간 |
|---|---|---|---|
| 1 | 3.64 | 3.50 | ~7분 |
| 2 | 3.40 | 3.49 | ~7분 |
| 3 | 3.26 | 3.51 | ~7분 |

- Train CE가 계속 감소하지만 Val CE는 epoch 2부터 미세 상승 (3.49→3.51) → **과적합 시작**
- WikiText-2가 작아서 3 epoch만에 FT가 학습 데이터에 과적합되기 시작
- KD는 과적합 없음 (Val CE 계속 감소) — Teacher soft label이 정규화(regularization) 역할

### 6. KD가 FT보다 PPL이 높은 이유 분석

현재 KD(48.96) > FT(32.56)인 4가지 원인:

**① Teacher-Student 용량 갭이 작음**
- Teacher(gpt2, 124M) vs Student(distilgpt2, 82M): 파라미터 차이 1.5배
- Teacher가 가진 "추가 지식"이 적어서 soft label의 정보량이 제한적
- 논문에서 KD 효과가 큰 경우: Teacher가 Student보다 **5~10배** 큰 경우

**② Temperature T=4.0이 너무 높음**
- T가 높을수록 softmax 출력이 uniform(평평)에 가까워짐
- T=4.0이면 Teacher의 출력 분포가 매우 flat → 각 토큰 간 확률 차이가 희석
- Teacher와 Student의 갭이 작은 상황에서 flat distribution은 **노이즈에 가까움**
- 결과: KD loss=99로 여전히 높음 → Student가 flat한 분포를 모방하느라 CE 학습이 방해받음
- 근거: Hinton(2015)은 T=2~5를 제안했으나, 이는 classification 기준. LM에서는 vocab이 50,257개로 크기 때문에 T를 낮춰야 의미 있는 분포 차이가 유지됨

**③ α=0.3으로 CE 비중이 낮음**
- 현재: Loss = 0.3·CE + 0.7·KD → KD(soft label) 비중이 70%
- Teacher가 "좋은 선생"이 아닌 상황에서 KD 비중이 높으면 성능 저하
- FT는 CE 100%로 hard label만 학습 → 작은 데이터에서는 이게 더 효율적
- α=0.5~0.7로 올리면 CE 비중이 높아져 hard label 학습이 강화됨

**④ WikiText-2가 작음 (2M 토큰)**
- 작은 데이터에서는 FT가 단순히 정답만 외워도 PPL이 낮아짐
- KD는 Teacher 분포 모방이라는 추가 목표가 있어서 수렴이 느림
- 더 큰 데이터(WikiText-103, 103M 토큰)에서는 FT가 외울 수 없으므로 KD의 정규화 효과가 빛남

### 7. 다음 실험 추천 및 근거

#### 추천 1: T=2.0~3.0으로 낮추기

**근거**: Teacher(gpt2) → Student(distilgpt2) 갭이 1.5배로 작음. T=4.0은 50,257개 vocab에 대해 분포를 너무 평탄하게 만들어서, Teacher가 "특정 토큰이 더 가능성 높다"는 정보가 희석됨.

T를 낮추면:
- Teacher 출력 분포가 sharp해져서 → "이 토큰이 정답은 아니지만 유력하다"는 dark knowledge가 명확히 전달
- KD loss가 더 빠르게 감소할 것
- 예상: T=2.0에서 KD PPL이 40 이하로 개선 가능

#### 추천 2: α=0.5~0.7로 올리기

**근거**: 현재 α=0.3이면 CE 30% + KD 70%. Teacher가 약한 상황에서 KD에 70%를 할당하면 효과적이지 않음.

α를 올리면:
- Hard label(정답) 학습 비중이 증가 → FT와 비슷한 효과 + KD의 정규화 효과까지
- α=0.5면 CE 50% + KD 50%: FT의 빠른 수렴 + KD의 과적합 방지를 동시에 기대
- 예상: α=0.7이면 KD PPL이 FT 수준(~32)에 근접하면서 과적합에는 더 강할 것

#### 추천 3: 더 큰 Teacher (gpt2-large, 774M)

**근거**: KD의 핵심은 Teacher가 Student보다 훨씬 뛰어나야 한다는 것. 현재 Teacher PPL(43.60)과 Student KD PPL(48.96)의 차이가 5.36밖에 안 됨 → Teacher가 가르칠 게 별로 없음.

gpt2-large를 쓰면:
- Teacher PPL ≈ 20~25 (WikiText-2 기준) → Student와의 갭이 20~40으로 벌어짐
- Teacher의 soft label에 담긴 정보량이 대폭 증가
- 맥북에서 epoch당 ~40~60분 소요 (gpt2의 ~4배)
- 예상: KD PPL이 FT를 이기기 시작할 가능성

#### 추천 4: WikiText-103 (103M 토큰)

**근거**: WikiText-2(2M)에서 FT가 3 epoch만에 과적합됨 (Val CE가 epoch 2부터 상승). 데이터가 50배 커지면:

- FT가 데이터를 암기할 수 없음 → 과적합 방지
- KD의 정규화(Teacher soft label) 효과가 드러남 → KD가 FT를 이길 가능성
- 단, epoch당 ~10시간 (맥북 기준) → **GPU 서버 필요**

#### 맥북에서 추천하는 다음 실험 순서

| 순서 | 실험 | 시간 | config 변경 |
|---|---|---|---|
| 1 | T=2.0, α=0.5 | ~36분 | temperature=2.0, alpha=0.5 |
| 2 | T=3.0, α=0.7 | ~36분 | temperature=3.0, alpha=0.7 |
| 3 | gpt2-medium Teacher | ~2시간 | teacher_model=gpt2-medium |

