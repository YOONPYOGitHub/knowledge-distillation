# 한국어 KD 3개 계열 비교 — 진행 현황

기준 시각: **2026-10-01 13:20 KST** · 브랜치: `feature/kanana-4way-kd` · 조모임(10/1 21:30) 공유용

Qwen2.5, Gemma 3, Kanana 1.5 세 계열에서 같은 한국어 위키 데이터로 Teacher → Student 지식증류(KD)를 4-way
(Teacher FT / Student KD / Student FT / Student Base)로 비교한다. 이 문서는 각 실험 문서의 숫자를 한곳에 모은 현황판이다.

## 한눈에 보기

1. **데이터가 1,000문서일 때는 KD 이득이 작거나 불확실했다.** 1만 문서로 늘린 Gemma v4에서 처음으로 KD가 FT를 크게, 통계적으로 확실하게 이겼다(−5.9%).
2. **세 계열을 같은 조건으로 놓을 수 있는 것은 아직 Gemma v4와 Kanana v4 둘뿐이다.** Qwen은 1,000문서·다른 KD 설정(v3)까지만 있다.
3. **Kanana v4는 Teacher FT와 Student SFT를 마치고 KD 진행 중이다.** KD 완료와 4-way 평가는 10/2 새벽 예정이다.
4. **3개 계열 공정 비교를 완성하려면 Qwen v4(1만 문서, Gemma·Kanana v4와 같은 KD 설정)가 필요하다.**

## 1. 비교 대상

| 계열 | Teacher → Student | 파라미터 (T / S) | 압축비 | 성격 |
|---|---|---:|---:|---|
| Qwen2.5 | 7B → 1.5B | 7.62B / 1.54B | 4.9배 | 다국어 범용 |
| Gemma 3 | 12B pt → 1B pt | 12.19B / 1.00B | 12.2배 | 다국어 범용 |
| Kanana 1.5 | 8B base → 2.1B base | 8.03B / 2.09B | 3.8배 | 한국어 특화 |

압축비가 계열마다 크게 다르다(3.8 ~ 12.2배). 통제하지 못한 변수로 남는다.

## 2. 실험 조건 비교

공통: 한국어 위키 `wikimedia/wikipedia 20231101.ko`(revision 고정), split seed 42, train/val/test 90/5/5, seq 256,
T = 2, α = 0.5, tokenmean, lr 1e-5, Adafactor, Teacher LoRA r16 3 epoch, best는 val CE 기준, seed 하나.

| 항목 | Qwen v3 | Gemma v4 | Kanana v4 |
|---|---|---|---|
| 데이터 | **1,000문서** | 10,000문서 | 10,000문서 |
| KD 초기값 | **SFT 체크포인트** | Base | Base |
| KD divergence | **reverse KL** | forward KL | forward KL |
| KD 목표 분포 | **full-vocab** | top-K 128 | top-K 128 |
| KD Teacher 정밀도 | bf16 | **int8** (메모리) | bf16 |
| BOS | 붙이지 않음 | 붙임 (Gemma 필수) | 붙이지 않음 (붙이면 PPL 악화) |
| epoch / early stopping | 8 / 없음 | 4 / patience 1 | 4 / patience 1 |
| 장비 | H100 NVL | RTX 3090 × 4 | RTX 3090 × 4 |
| KD vs FT 통계 검정 | 안 함 | paired bootstrap | 평가 후 실행 예정 |

**Gemma v4와 Kanana v4는 같은 프로토콜**이다(차이는 모델별 입력 형식인 BOS와, 메모리 때문에 생긴 Gemma Teacher int8뿐).
Qwen v3는 데이터 양과 KD 설정이 모두 달라 두 실험과 나란히 놓을 수 없다.

## 3. 결과

### 지표 읽는 법

- **PPL은 같은 계열 안에서만 비교한다.** tokenizer가 다르면 같은 문장도 토큰 수가 달라 PPL 절대값의 의미가 다르다.
- 계열 간 비교는 단위가 없는 상대 지표로 한다.
  - **FT → KD 개선율** = (FT − KD) / FT
  - **격차 중 KD가 좁힌 비율** = (FT − KD) / (FT − Teacher). Teacher와 FT 사이 간격의 몇 %를 KD가 메웠는지.
- **KD vs FT 차이가 노이즈를 넘는지는 paired bootstrap으로 판정한다.** 같은 test chunk에서 10,000회 복원추출한 95% 신뢰구간(CI)이 0을 포함하지 않아야 "유의"다.
  실험 자동 요약(`notes.md`)은 KD < FT이면 무조건 "증류 효과 확인됨"으로 적으므로 판정 근거로 쓰지 않는다.

### 3.1 1,000문서 실험

| 실험 | Teacher (FT) | KD | FT | Base | FT → KD | 격차 중 KD 몫 | 통계 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen v3 (SFT 초기값 + reverse, α 0.5) | 7.657 | 11.685 | 11.814 | 12.127 | −1.09% | 3.1% | 검정 안 함 |
| Qwen H100 튜닝 (SFT 초기값 + reverse, α 0.7)¹ | 7.657 | 11.353 | 11.815 | — | −3.9% | 11.1% | 검정 안 함 |
| Gemma v3 (SFT 초기값 + reverse, top-K) | 6.884 | 11.643 | 11.663 | 14.004 | −0.17% | 0.4% | **유의하지 않음** (CI [−0.148, +0.118]) |
| Gemma v3 요인 분리 (Base 초기값 + forward) | 6.884 | 11.457 | 11.663 | 14.004 | −1.77% | 4.3% | **유의** (CI [−0.320, −0.080]) |

¹ 9/20 H100 하이퍼파라미터 탐색(lr, α, T)에서 validation으로 고른 설정의 held-out test 결과. Teacher와 SFT 초기값은 Qwen v3와 같다.
test 35,955 토큰으로 작고 seed 하나다.
[튜닝 보고서](https://github.com/YOONPYOGitHub/knowledge-distillation/blob/feature/azure-h100-kd-tuning/results/logs/h100-tuning-20260920/report.md)

- 1,000문서에서는 KD 설정에 따라 결과가 크게 갈렸다. Gemma에서는 **SFT 체크포인트에서 KD를 시작하면 이득이 사라지고, Base에서 시작해야 유의한 이득**이 났다.
- Qwen은 α를 0.5 → 0.7로 올리자 이득이 1.1% → 3.9%로 커졌지만, 두 결과 모두 통계 검정을 하지 않았다.

### 3.2 10,000문서 실험

#### Gemma v4 — 완료 (9/27)

| 모델 | PPL |
|---|---:|
| Teacher (FT) | 5.418 |
| **Student (KD)** | **7.866** |
| Student (FT) | 8.363 |
| Student (Base) | 12.253 |

- **KD vs FT −0.496 (−5.9%)**, 95% CI [−0.535, −0.459], test chunk 832개 중 724개(87%)에서 KD가 낫다.
- **격차 중 KD가 좁힌 비율 16.9%.** 같은 KD 설정의 1,000문서(4.3%)보다 약 4배 크다. 데이터가 늘수록 KD 몫이 커진다.
- SFT는 데이터 10배에서도 epoch 1 이후 과적합(val CE 2.2428 → 2.2942)했지만, KD는 epoch 2까지 계속 개선(2.2065 → 2.1761)했다. **KD가 과적합을 막아 같은 데이터를 더 오래 유효하게 학습하게 한다.**

#### Kanana v4 — KD 진행 중

| 단계 | val CE (epoch별) | best | epoch당 시간 | 상태 |
|---|---|---|---:|---|
| Teacher FT (QLoRA nf4) | 1.7985 → 1.7842 → 1.7801 | epoch 2 ² | 약 109분 | ✅ 10/1 06:10 |
| Student SFT | **2.0541** → 2.1345 | epoch 1 | 약 315분 | ✅ 10/1 11:14 (epoch 2 과적합, early stopping) |
| KD (Base 초기값 + forward) | 진행 중 | — | 약 320분 | 🔄 10/1 13:01 시작 |
| 4-way 평가 + bootstrap | — | — | — | ⏳ KD 직후 자동 실행 |

² epoch 3의 val CE가 가장 낮지만 개선폭 0.0041이 기준(min_delta 0.005)에 못 미쳐 best로 저장되지 않았다.

- **SFT는 Gemma v4와 같은 양상이다.** epoch 1 이후 바로 과적합한다.
- 예상 일정(KST): KD epoch 1 완료 10/1 18:25 → epoch 2 23:50 → epoch 3 10/2 05:15 → 평가·GitHub push 10/2 05:30 전후.
  epoch 2 또는 3에서 early stopping이 걸리면 더 일찍 끝난다.
- **KD epoch 1의 val CE가 SFT best 2.0541보다 낮으면** Gemma v4와 같은 경향이다(Gemma v4는 KD epoch 1에서 이미 2.2065로 SFT best 2.2428보다 낮았다).
  val CE는 같은 split·같은 tokenizer라 계열 안에서는 바로 비교할 수 있다.
- 기준 평가(1,000문서 test, 학습 전): Teacher 8B 7.029, Student 2.1B 8.748. Teacher와 Student의 격차가 다른 계열보다 훨씬 작아(Gemma pt 8.214 vs 14.004),
  KD가 좁힐 수 있는 간격 자체가 작을 수 있다.

## 4. 현재까지의 해석

1. **KD 이득을 좌우한 것은 데이터 규모와 KD 시작점이었다.** 1,000문서에서는 SFT가 이미 대부분의 이득을 가져가고 KD 몫이 작았다.
   Base에서 시작하는 KD를 1만 문서로 돌리자 이득이 분명해졌다(Gemma).
2. **KD는 "더 오래 학습"이 아니라 "과적합 방지"로 이득을 낸다.** 같은 epoch에서 CE만 학습한 SFT는 나빠지고 KD는 좋아졌다.
3. Kanana에서도 같은 결과가 나오면 다국어(Gemma)와 한국어 특화(Kanana) 계열 모두에서 성립한다는 근거가 된다.

## 5. 남은 일과 논의할 것

| 항목 | 내용 | 비고 |
|---|---|---|
| **Qwen v4** | 1만 문서, Base 초기값 + forward KL, top-K 128로 Gemma·Kanana v4와 같은 조건 실행 | 3개 계열 공정 비교의 필수 조건. 담당·장비·일정 논의 필요 |
| Kanana v4 완료 | KD → 4-way 평가 → bootstrap → 결과 문서 | 10/2 새벽 자동 진행, 자동 commit·push |
| seed 반복 | 모든 실험이 seed 42 하나 | 최소 3회 권장. 실험당 약 18~30시간(3090 × 4) |
| 계열 간 절대 비교 지표 | bits-per-byte 추가 | tokenizer와 무관한 지표. 현재 미측정 |
| 생성 품질 | 한국어 생성 결과 정성 비교 | PPL만으로는 답변 품질을 보여주기 어렵다 |

### 운영 메모

- VESSL 워크스페이스는 최대 24시간이라 1만 문서 실험은 한 번은 재시작이 필요하다. **반드시 같은 워크스페이스를 재시작**해야 `/root`(체크포인트)가 남는다. 러너는 epoch 단위로 이어간다.
- 10/1 07:56 서버 호스트의 시스템 업데이트 뒤 컨테이너에서 GPU가 보이지 않는 문제(`Failed to initialize NVML: Unknown Error`)가 있었다. 워크스페이스 재시작으로 해결했다. GPU 센터에 공유 필요.
- 원격 작업은 tmux 안에서 실행하고, 결과는 단계가 끝날 때마다 서버에서 자동 commit·push한다.

## 참고 문서

| 문서 | 내용 |
|---|---|
| [docs/gemma3-4way-topk-kd.md](gemma3-4way-topk-kd.md) | Gemma v3 4-way, top-K vs full-vocab, bootstrap |
| [docs/gemma3-v3-ablation.md](gemma3-v3-ablation.md) | Gemma v3 KD 요인 분리 (초기값 × divergence) |
| [docs/gemma3-v4.md](gemma3-v4.md) | Gemma v4 (1만 문서) |
| [results/logs/qwen7b-korean-reverse-v3/notes.md](../results/logs/qwen7b-korean-reverse-v3/notes.md) | Qwen v3 |
| [results/kanana/logs/kanana-8b-2b-v4/](../results/kanana/logs/kanana-8b-2b-v4/) | Kanana v4 학습 이력과 로그 |
