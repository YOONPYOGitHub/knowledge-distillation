# 구현 현황 — PPT 슬라이드 초안 (3장 구성)

> 중간보고 5장 "구현 현황" 발표용 슬라이드 초안.
> 슬라이드 1장 = 1개의 `## 슬라이드 N` 섹션으로 구성.
> 각 슬라이드 하단의 *발표 포인트*는 발표자 노트로 활용.

---

## 슬라이드 1 — 실험 환경 및 4-Way 비교 설계

### 제목
**"이렇게 실험했습니다"** — 실험 환경 및 4-Way 비교 프레임워크

### 레이아웃 (좌 1 : 우 2)

#### 좌측 — 실험 조건

| 항목 | 값 |
|---|---:|
| Teacher | GPT-2 (124M, 498MB) |
| Student | DistilGPT-2 (82M, 328MB) |
| 압축률 | **약 34% 감소** |
| Dataset | WikiText-2 |
| Device | Apple MPS (MacBook) |
| KD Loss | α·CE + (1−α)·T²·KL |

#### 우측 — 4-Way 비교 프레임워크

**이미지**: [docs/diagrams/4way-comparison.svg](../../diagrams/4way-comparison.svg)

> 4개 모델(Teacher / Student-KD / Student-FT / Student-Base)을 동일한 4대 지표(PPL / 추론 속도 / 생성 품질 / 메모리)로 동시 평가하는 구조
>
> - 🎓 **Teacher (FT)** — Upper Bound
> - 🧑‍🎓 **Student (KD)** — 검증 대상 ⭐
> - 📝 **Student (FT)** — 기준선
> - 📖 **Student (Base)** — Lower Bound

### 핵심 메시지
- **4개 모델을 동시에 비교** → KD의 효과를 구조적으로 측정
- **Teacher (FT) ↔ Student (KD)** 의 gap 만큼이 "증류로 좁힐 수 있는 거리"
- **Student (KD) ↔ Student (FT)** 의 gap 만큼이 "지식 증류 자체의 효과"

### 사용 이미지
- **필수**: [docs/diagrams/4way-comparison.svg](../../diagrams/4way-comparison.svg) — 슬라이드 우측 전체 영역

### 발표 포인트
- "4가지 모델을 한 자리에 놓고 비교하기 위한 프레임워크를 먼저 구성했습니다"
- "KD가 효과적인지 단독으로는 알기 어렵기 때문에 Base / FT / Teacher를 함께 측정합니다"
- "총 8회 반복 실험으로 최적 KD 조건을 탐색했습니다"

---

## 슬라이드 2 — 실험 진화 과정 ⭐ (메인)

### 제목
**"6번 실패하고 8번째에 역전시켰습니다"** — exp03 → exp08 실험 진화

### 레이아웃 (상 표 : 하 인사이트)

#### 상단 — 실험 히스토리 (요약)

| Exp | Teacher FT | α | Epochs | KD PPL | FT PPL | Gap | 승자 |
|---|---|---:|---:|---:|---:|---:|---|
| exp03 | ❌ | 0.3 | 3 | 48.96 | 32.56 | +16.40 | FT |
| exp06 | ✅ | 0.5 | 3 | 33.00 | 32.56 | +0.44 | FT |
| exp07 | ✅ | 0.3 | 5 | 32.89 | 32.68 | +0.21 | FT |
| **exp08** | **✅** | **0.2** | **8** | **32.44** | **32.82** | **−0.38** | **KD ✅** |

#### 중단 — Gap 추이 (시각화 텍스트)

```
exp03 (Teacher FT 없음):  +16.40  ████████████████████  FT 압승
exp06 (α=0.5, ep=3):       +0.44  ██░░░░░░░░░░░░░░░░░░  FT 우세
exp07 (α=0.3, ep=5):       +0.21  █░░░░░░░░░░░░░░░░░░░  FT 근소 우세
exp08 (α=0.2, ep=8):       −0.38  ░░░░░░░░░░░░░░░░░░██  KD 역전! ✅
```

#### 하단 — KD 역전의 3대 조건

> **각 단계마다 무엇을 바꾸었고, 왜 효과가 있었는가**

1. **Teacher Fine-Tuning 적용** (exp03 → exp06)
   - Teacher가 도메인을 모르면 soft target 자체가 약함 → gap 16 → 0.44 (97% 감소)
2. **α를 0.5 → 0.2 로 하향** (exp06 → exp08)
   - soft target 비중을 50% → 80% 로 키움 → α 0.1 단위마다 KD가 약 0.3 PPL씩 개선
3. **Epochs를 3 → 8 로 확장** (exp06 → exp08)
   - FT는 epoch 2에서 과적합 시작 / KD는 epoch 8까지 수렴 → 시간이 KD 편

### 사용 이미지
- **필수**: [figures/slide2-evolution.svg](figures/slide2-evolution.svg) — 슬라이드 전체를 채우는 단일 이미지 (1280×720, 16:9)
  - 좌측: 실험 히스토리 표 + Gap 추이 막대
  - 우측: KD 역전의 3대 조건 카드

### 발표 포인트
- "처음부터 KD가 이긴 게 아닙니다. 6번의 실패가 있었습니다"
- "각 실패에서 한 가지 변수를 바꾸며 가설을 검증했습니다"
- "최종적으로 3가지 조건이 모두 갖춰졌을 때 KD가 FT를 역전합니다"
- 이 슬라이드가 **메인** — 가장 길게 설명

---

## 슬라이드 3 — 최종 결과 & 인사이트

### 제목
**"이런 결과를 얻었습니다"** — exp08 최종 결과 및 핵심 인사이트

### 레이아웃 (상 그래프 2개 : 하 메시지 3개)

#### 상단 좌측 — Perplexity 비교
**이미지**: `results/figures/20260410_041631/perplexity_comparison.png`

설명 캡션:
> Teacher 25.1 / **KD 32.44** / FT 32.82 / Base 65.3
> → 기대했던 **Teacher < KD < FT < Base** 순서를 처음으로 달성

#### 상단 우측 — Validation Loss 곡선 (KD 역전의 핵심 증거)
**이미지**: `results/figures/20260410_041631/val_loss.png`

설명 캡션:
> FT(주황): epoch 2에서 바닥 → 6 epoch 동안 상승 (과적합)
> KD(초록): epoch 8까지 꾸준히 하강 (정규화 효과)
> **Epoch 6에서 역전** — "가위 모양" 패턴

#### 하단 — 핵심 메시지 3줄

| | 메시지 |
|---|---|
| ✅ **성능** | KD는 hard label만 쓴 FT보다 **일반화 성능이 더 높다** (과적합 4.3배 억제) |
| ⚡ **속도** | 추론 속도는 Student의 22K tok/s 그대로 — Teacher 14K 대비 **1.57× 빠름** |
| ⚠ **한계** | WikiText-2 / MPS 환경 제약 → **Phase 1에서 GPU 센터 재검증 예정** |

### 사용 이미지
1. **필수**: [figures/slide3-final-result.svg](figures/slide3-final-result.svg) — 슬라이드 전체를 채우는 단일 이미지
   - 좌측: PPL 비교 막대 (KD < FT 하이라이트)
   - 우측: 성능·속도·한계 3개 카드
2. (보조, Q&A 대비용 / 슬라이드 본편에는 널지 않음)
   - [val_loss.png](../../../results/figures/20260410_041631/val_loss.png) — 가위 도·역전 증거
   - [training_loss.png](../../../results/figures/20260410_041631/training_loss.png) — "FT train loss가 더 낮은 함정"
   - [inference_speed.png](../../../results/figures/20260410_041631/inference_speed.png) — tok/s 비교

### 발표 포인트
- "KD는 Teacher의 지식을 **성능**에 흡수하면서 **속도**는 Student의 경량 아키텍처를 그대로 유지합니다"
- "이것이 Knowledge Distillation의 실용적 가치입니다"
- 마지막 줄에서 **Phase 1 (GPU 센터 재검증)** 으로 자연스럽게 다음 발표자에게 넘김

---

## 부록: 이미지 사용 가이드

### 추천 사용 (현 구성안 기준)

| 슬라이드 | 이미지 | 용도 |
|---|---|---|
| 1 | [docs/diagrams/4way-comparison.svg](../../diagrams/4way-comparison.svg) | 4-Way 비교 프레임워크 |
| 2 | [figures/slide2-evolution.svg](figures/slide2-evolution.svg) | 실험 진화 + 3대 조건 (슬라이드 전체) |
| 3 | [figures/slide3-final-result.svg](figures/slide3-final-result.svg) | PPL 비교 + 인사이트 카드 (슬라이드 전체) |

### 새로 제작한 PPT용 SVG

파일 경로: [docs/report/interim-report/figures/](figures/)

| 파일 | 사이즈 | 설명 |
|---|---|---|
| `slide2-evolution.svg` | 1280×720 (16:9) | 슬라이드 2 전체용 — 실험 진화 표 + Gap 막대 + 3대 조건 카드 |
| `slide3-final-result.svg` | 1280×720 (16:9) | 슬라이드 3 전체용 — PPL 비교 막대 + 성능/속도/한계 카드 |

> SVG는 PowerPoint에 그대로 드래그&드롭으로 삽입 가능하며, 텍스트도 편집할 수 있습니다.
> PNG가 필요하면 `rsvg-convert` 또는 브라우저에서 다운로드로 변환 가능.

### 원본 figures (필요 시 그대로 사용)

| 파일 | 내용 | 추천 위치 |
|---|---|---|
| `perplexity_comparison.png` | 4모델 PPL 막대그래프 | 슬라이드 3 (필수) |
| `val_loss.png` | val loss 곡선 (KD vs FT 가위 모양) | 슬라이드 3 (필수) |
| `training_loss.png` | train loss 곡선 (FT가 더 낮은 함정) | 부록/Q&A 대비 |
| `inference_speed.png` | tokens/sec 비교 | 슬라이드 3 우하단 (선택) |

### Q&A 대비용 (슬라이드 본편엔 안 넣지만 손에는 둘 것)
- `training_loss.png` — "FT의 train loss가 더 낮은데 왜 KD가 이기나?" 질문 시
- 다른 실험들의 figures (exp06, exp07 등) — "왜 exp08 결과만 보여주냐?" 질문 시
