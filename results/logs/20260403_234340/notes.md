# 실험 노트 — 20260403_234340

## 자동 진단

- ⚠️ Teacher PPL=32574 — 비정상적으로 높음 (패딩 과다 또는 미학습)
- ⚠️ Student (FT) PPL=1.00 — 과적합 의심 (패딩 토큰 예측으로 loss 왜곡 가능)
- ⚠️ Student (Base) PPL=36266 — 비정상적으로 높음 (패딩 과다 또는 미학습)
- ⚠️ KD(128.22) >= FT(1.00) — 증류 효과 미확인

## 다음 액션

- [ ] temperature 값 조정 (현재 → +2 시도)
- [ ] alpha 값 낮추기 (CE 비율 줄이기)
- [ ] GPU 서버에서 server_config()로 본 실험 수행

## 메모

### 1. 실험 조건 요약

| 항목 | 값 | exp01 대비 변경 |
|---|---|---|
| Teacher | gpt2 (124M, 497.8MB) | - |
| Student | distilgpt2 (82M, 327.7MB) | - |
| 압축률 | 파라미터 34% 감소, 용량 34% 감소 | - |
| Temperature | 4.0 | 3.0 → **4.0** |
| Alpha (α) | 0.3 → KD Loss = 0.3·CE + 0.7·T²·KL | 0.5 → **0.3** |
| Epochs | 3 | 1 → **3** |
| Batch Size | 4 | 2 → **4** |
| max_seq_length | 256 | 128 → **256** |
| Device | Apple MPS (MacBook) | - |

### 2. PPL 수치 분석

| 모델 | PPL | avg_loss | 해석 |
|---|---|---|---|
| **Teacher (gpt2)** | 32,574 | 10.39 | exp01(20,803)보다 높음 — seq=256에서 패딩 비율은 줄었지만 보는 문맥이 길어 Teacher PPL 상승 |
| **Student (KD)** | 128.22 | 4.85 | exp01(57.08)보다 높지만, 더 긴 문맥에서의 수치이므로 직접 비교 어려움 |
| **Student (FT)** | **1.00** | 0.00006 | 여전히 과적합 — seq=256에서도 패딩 패턴 암기 지속 |
| **Student (Base)** | 36,266 | 10.50 | 미학습 distilgpt2, Teacher와 비슷한 수준 |

> **핵심**: seq=256로 올려도 FT PPL=1.0 문제는 해소되지 않았다.
> 패딩 왜곡은 seq_length가 아니라 **평가 코드의 label 처리 방식** 개선이 필요할 수 있다.

### 3. KD 학습 수렴 추이 (3 Epochs)

| Epoch | Train CE | Train KD | Train Total | Val CE | Val Total | 시간 |
|---|---|---|---|---|---|---|
| 1 | 5.26 | 48.93 | 35.83 | 5.14 | 25.87 | ~30분 |
| 2 | 4.99 | 43.28 | 31.80 | 4.90 | 25.59 | ~30분 |
| 3 | 4.91 | 40.93 | 30.13 | 4.88 | 23.95 | ~30분 |

**관찰:**
- KD Loss가 **epoch마다 꾸준히 감소** (48.9 → 43.3 → 40.9, 총 -16.3%) → Teacher 모방 개선 중
- Val CE가 **epoch 2→3에서 0.02만 감소** (4.90 → 4.88) → 수렴에 가까워짐
- Train/Val CE 차이가 거의 없음 (4.91 vs 4.88) → **과적합 없음**
- 총 학습 시간: ~90분 (30분 × 3 epoch)

### 4. FT 학습 결과

| Epoch | Train CE | Val CE | 시간 |
|---|---|---|---|
| 1 | 0.0129 | 0.00005 | ~19분 |
| 2 | 0.0003 | 0.00001 | ~18분 |
| 3 | 0.0001 | 0.00006 | ~18분 |

**관찰:**
- 1 epoch만에 CE가 0.01 이하 → 패딩 패턴을 즉시 학습
- Epoch 3에서 Val CE가 미세하게 상승 (0.00001 → 0.00006) → 과적합 시작 신호
- KD 대비 학습 시간 37% 짧음 (18분 vs 30분) → Teacher forward pass 오버헤드 없음

### 5. 추론 속도 비교

| 모델 | Tokens/sec | ms/token | 모델 크기 |
|---|---|---|---|
| Teacher (gpt2) | 5,919 | 0.169 | 497.8 MB |
| Student (KD) | 9,301 | 0.108 | 327.7 MB |
| Student (FT) | 9,307 | 0.107 | 327.7 MB |
| Student (Base) | 9,231 | 0.108 | 327.7 MB |

**관찰:**
- Student 모델들은 Teacher 대비 **약 1.57배 빠름** (9,301 / 5,919)
- exp01(1.49배)보다 속도 격차가 약간 벌어짐 — seq=256에서 Teacher 오버헤드가 더 크게 작용
- Student 3종 간 속도 차이는 76 tok/s 이내 → 아키텍처가 같으면 속도 동일

### 6. exp01 대비 변경 효과

| 변경 사항 | 기대 효과 | 실제 결과 |
|---|---|---|
| seq 128→256 | 패딩 비율 감소 → PPL 왜곡 완화 | ❌ FT PPL=1.0 여전 — 패딩 왜곡 미해소 |
| T 3.0→4.0 | softer distribution → KD 학습 용이 | △ KD PPL이 더 높지만 seq 차이로 판단 불가 |
| α 0.5→0.3 | KL 비중 높여 Teacher 지식 더 반영 | ✅ KD Loss 감소폭 양호 (3 epoch에서 -16%) |
| epochs 1→3 | 수렴 추이 확인 | ✅ 수렴 확인 — Val CE 감소폭이 점점 줄어듦 |
| batch 2→4 | 학습 안정성 향상 | ✅ 안정적 학습, 과적합 없음 |

### 7. 서버 본실험에서 바꿔야 할 것

| 항목 | 현재 (exp02) | 서버 목표 | 이유 |
|---|---|---|---|
| max_seq_length | 256 | **512** | 패딩 비율 추가 감소 |
| epochs | 3 | **3~5** | 이미 수렴 추세이므로 3이면 충분할 수 있음 |
| batch_size | 4 | **8~16** | GPU 메모리 활용 극대화 |
| teacher | gpt2 (124M) | **gpt2-large (774M)** | Teacher-Student 용량 갭 확대 → KD 효과 극대화 |
| student | distilgpt2 (82M) | **gpt2 (124M)** | 더 큰 Student로 지식 수용 능력 확보 |
| 평가 방식 | 패딩 포함 | **패딩 제외 PPL** | label=-100 처리 검증 필요 |

### 8. 결론

> **exp02는 "수렴 확인 + 설정 변경 효과 검증"이 목적이었고, 부분적으로 달성.**
> - ✅ 3 epoch에서 수렴 추세 확인 (Val CE 감소폭 축소)
> - ✅ α=0.3 설정으로 KD Loss 안정적 감소
> - ❌ seq=256로는 FT 과적합 문제 미해소
> - 추론 속도 비교(Teacher 5,919 vs Student 9,301 tok/s, **1.57배**)는 유효한 결과
> - **평가 코드의 패딩 처리 방식 점검이 다음 우선 과제**

---

### 9. exp02 결과 기반 코드 수정 사항

exp01~02의 결과를 분석한 결과, config 튜닝 이전에 **코드 수준의 구조적 문제**가 발견됨.
PPL 수치의 신뢰성을 확보하기 위해 아래 수정을 진행한다.

#### 문제 A: 데이터 전처리 — Padding 방식 → Packing 방식

**증거**: Teacher(GPT-2)의 PPL이 32,574로, 논문 기준 29.41과 1,000배 이상 괴리.
Student(FT) PPL=1.00은 과적합이 아니라 패딩 토큰 암기.

- 현재: 문장별로 `padding="max_length"` → 짧은 문장에 대량의 패딩 토큰 발생
- 수정: 전체 텍스트를 연결한 뒤 `max_seq_length` 단위로 chunking (Packing)
- 효과: 패딩 토큰이 원천적으로 제거됨. GPT-2 논문, DistilGPT-2, MiniLLM 등 모든 LM 논문의 표준 방식

#### 문제 B: Causal LM Label Shift 누락

**증거**: PPL 수치가 논문과 비교 불가능한 값. 자기 자신을 예측하는 것은 next-token prediction이 아님.

- 현재: `F.cross_entropy(logits, labels)` — logits[t]가 labels[t](자기 자신)을 예측
- 수정: `logits[..., :-1, :]` → `labels[..., 1:]` 로 1칸 shift
- 효과: HuggingFace `GPT2LMHeadModel` 내부 구현과 동일. PPL이 논문 기준의 절대적 값이 됨
- 적용 범위: `distill.py`, `train_baseline.py`, `evaluate.py` 모두

#### 문제 C: KD Loss 패딩 마스킹 없음

**증거**: KD loss가 48.9→40.9로 감소했지만, 패딩 위치에서의 Teacher-Student 분포 매칭이 포함된 수치.

- 현재: KL divergence를 모든 토큰 위치(패딩 포함)에서 계산
- 수정: Packing 적용 시 패딩이 없으므로 자동 해결
- 참고: Packing 미적용 환경에서는 `attention_mask`로 유효 토큰만 필터링 필요

#### 참고 논문

| 논문 | 관련 수정 |
|---|---|
| Hinton et al. (2015) "Distilling the Knowledge in a Neural Network" | KD loss 구조 (현재 코드가 따르는 원조) |
| Sanh et al. (2019) "DistilBERT" | attention_mask로 패딩 제외 후 loss 계산 |
| Gu et al. (2023) "MiniLLM" (ICLR 2024) | Packing + label shift + reverse KL 사용 |
| Xu et al. (2024) "A Survey on KD of LLMs" | white-box KD의 표준 파이프라인 정리 |

#### 수정 후 기대 효과

| 지표 | 현재 (exp02) | 수정 후 기대 |
|---|---|---|
| Teacher PPL | 32,574 | ~29~35 (GPT-2 논문 기준) |
| Student (FT) PPL | 1.00 (패딩 암기) | 정상적인 값 (KD와 비교 가능) |
| Student (KD) PPL | 128.22 (의미 불명) | 논문 비교 가능한 절대 수치 |
| KD vs FT 비교 | 왜곡 (FT가 항상 이김) | 공정 비교 가능 |

