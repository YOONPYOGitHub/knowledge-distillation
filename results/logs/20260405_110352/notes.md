# 실험 노트 — 20260405_110352 (exp05: WikiText-103)

## 실험 설정

| 항목 | 값 |
|------|-----|
| Teacher | gpt2-medium (355M) |
| Student | distilgpt2 (82M) |
| Dataset | **WikiText-103** (~103M tokens) |
| Temperature | 2.0 |
| Alpha | 0.5 |
| Epochs | 3 |
| Batch Size | 4 |
| Max Seq Length | 256 |
| Device | MPS (Apple Silicon) |

## 결과 요약

| 모델 | Perplexity (↓) | Tokens/sec (↑) | Size |
|------|---------------|----------------|------|
| Teacher (gpt2-medium) | 31.32 | 5,252 | 1,419 MB |
| **Student (FT)** | **22.72** | 22,343 | 328 MB |
| Student (KD) | 40.96 | 22,207 | 328 MB |
| Student (Base) | 65.27 | 22,305 | 328 MB |

## 학습 이력

### KD (에폭당 ~13.2시간, 총 ~39.8시간)

| Epoch | Train CE | Val CE | KD Loss | Time |
|-------|----------|--------|---------|------|
| 1 | 3.920 | 3.765 | 195.22 | 47,679s |
| 2 | 3.862 | 3.742 | 173.29 | 47,811s |
| 3 | 3.841 | 3.733 | 165.28 | 47,630s |

### SFT (에폭당 ~5.9시간, 총 ~17.8시간)

| Epoch | Train CE | Val CE | Time |
|-------|----------|--------|------|
| 1 | 3.402 | 3.192 | 21,462s |
| 2 | 3.265 | 3.139 | 21,407s |
| 3 | 3.211 | 3.114 | 21,395s |

## 자동 진단

- ⚠️ KD(PPL 40.96) >> FT(PPL 22.72) — 증류 효과 미확인
- ⚠️ FT가 teacher(PPL 31.32)보다도 낮은 PPL 달성 — student가 teacher를 능가

## 핵심 분석

1. **SFT가 KD를 압도**: FT(22.72) vs KD(40.96), PPL 차이 약 1.8배
2. **Student > Teacher**: FT student(22.72)가 teacher(31.32)보다 우수 — 데이터 충분 시 작은 모델도 특정 도메인에서 큰 모델 능가 가능
3. **KD의 soft target이 오히려 방해**: α=0.5에서 gradient의 절반이 teacher soft label 모방에 쓰여, 직접 학습 대비 비효율적
4. **시간 효율도 SFT 우위**: KD 총 39.8시간 vs SFT 총 17.8시간 (KD가 2.2배 더 소요)

## 전체 실험 누적 비교 (유효 실험만)

| Run | Dataset | Teacher | T/α | KD PPL | FT PPL | KD > FT? |
|-----|---------|---------|-----|--------|--------|----------|
| exp03 (0404_082410) | WikiText-2 | gpt2 | 4.0/0.3 | 48.96 | 32.56 | ⚠️ KD 열세 |
| exp04 (0404_223130) | WikiText-2 | gpt2 | 2.0/0.5 | 48.71 | 32.69 | ⚠️ KD 열세 |
| exp05 (0405_093536) | WikiText-2 | gpt2-medium | 2.0/0.5 | 47.31 | 32.77 | ⚠️ KD 열세 |
| **exp06 (0405_110352)** | **WikiText-103** | **gpt2-medium** | **2.0/0.5** | **40.96** | **22.72** | **⚠️ KD 열세** |

→ **6회 실험 중 KD가 FT를 이긴 적 없음**

## KD가 구조적으로 지는 근본 원인 분석

### 원인 1: Teacher가 fine-tune 되지 않은 pretrained 상태
- gpt2-medium은 WebText에서 pretrained된 상태 그대로 WikiText에 투입
- Teacher PPL(31.32) > FT Student PPL(22.72) → **teacher의 soft target이 ground truth보다 나쁨**
- 열등한 teacher로부터 배우면 성능이 떨어지는 건 당연

### 원인 2: distilgpt2는 이미 gpt2에서 증류된 모델
- HuggingFace가 gpt2로부터 KD해서 만든 모델
- gpt2 계열의 dark knowledge가 이미 내재 → 같은 teacher로 다시 KD해도 **추가 이득 없음**

### 원인 3: 데이터가 충분
- WikiText-103 ~103M tokens, student 82M params → 데이터/파라미터 비율 충분
- KD는 데이터가 부족할 때 이점이 크지만, 충분하면 직접 학습이 우세

### 원인 4: α=0.5로 gradient의 절반 낭비
- 열등한 teacher의 soft target을 따라가는 데 gradient 50% 소모

### 코드 레벨 확인 사항
- KD 구현 자체는 정확 (Hinton 2015 forward KL, T² 보정, label shift, teacher freeze 모두 올바름)
- KD vs FT 비교도 공정 (동일 데이터, 동일 옵티마이저, 동일 하이퍼파라미터)
- `warmup_steps=100`이 config에 정의되어 있지만 **실제 사용되지 않음** → scheduler 미구현

## KD가 working하기 위한 조건

| 조건 | 현재 | 수정 방향 |
|------|------|----------|
| Teacher 품질 | Pretrained 그대로 (PPL 31.32) | **Teacher를 먼저 fine-tune** → PPL을 student FT보다 낮게 |
| Student 모델 | distilgpt2 (이미 gpt2 증류) | **gpt2 계열과 무관한 모델** (Pythia-70M 등) |
| 데이터 크기 | WikiText-103 (충분) | **소량 데이터**에서 KD 이점 먼저 확인 |
| α 값 | 0.5 고정 | **α 탐색** (teacher 강할수록 α 낮게) |
| LR scheduler | 미구현 | **Warmup + cosine decay** 추가 |

## 다음 액션 후보 (우선순위순)

1. [ ] **Teacher fine-tune 후 KD** — gpt2-medium을 WikiText에 FT → 이 teacher로 KD (가장 임팩트 큼)
2. [ ] α 값 탐색 (0.1, 0.3, 0.7, 0.9) — 현재 α=0.5가 최적이 아닐 가능성
3. [ ] LR scheduler 구현 — warmup + cosine decay 적용
4. [ ] 데이터 크기별 비교 (WikiText-2 vs 103 10% vs 103 전체) — KD 우위 임계점 탐색
5. [ ] Student 모델 교체 (Pythia-70M 등) — gpt2 계열 중복 제거
6. [ ] OOD 평가 (PTB, LAMBADA) — KD의 일반화 이점 검증

