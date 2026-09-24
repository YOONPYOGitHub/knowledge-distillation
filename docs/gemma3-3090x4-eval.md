# Gemma 3 12B → 1B: 3090 × 4 기준선 평가

측정일: 2026-09-22. 브랜치: `feature/gemma3-12b-1b-3090x4-eval`.
환경: RTX 3090 24GB × 4, PyTorch 2.3.1+cu121, transformers 4.51.3, bitsandbytes 0.45.5.

> 후속: 이 기준선 위에서 4-way KD를 끝까지 돌린 결과는 [gemma3-4way-topk-kd.md](gemma3-4way-topk-kd.md)에 있다.

Qwen 실험과 **동일한 지표**(Perplexity / tokens·sec / ms·token / 파라미터)와 **동일한 데이터 조건**
(한국어 위키 `20231101.ko`, revision 고정, 1,000문서, seq 256, split seed 42)으로 측정했다.

## 결과

Teacher는 bf16으로 GPU 2장에 분할, Student는 cuda:2. test split은 118 chunk × 256 = 30,208 토큰.

| 모델 | 파라미터 | PPL | tokens/s | ms/token |
|---|---:|---:|---:|---:|
| Teacher (Gemma 3 12B pt) | 12,187,325,040 | **8.214** | 1,641 | 0.609 |
| Student Base (Gemma 3 1B pt) | 999,885,952 | **14.006** | 7,214 | 0.139 |

Teacher 파라미터는 텍스트 11,770,459,008 + vision tower 416,866,032로, 텍스트 평가에 쓰이지 않는
vision tower가 3.4%를 차지한다. `evaluation_results.json`에 분리해 기록한다.

증류 전 기준선: Teacher가 Student보다 PPL **5.79 낮고**, Student는 **4.4배 빠르다**.

## ⚠️ BOS 없이 측정하면 결과가 뒤집힌다

기존 패킹은 문서를 이어붙여 256토큰으로 자르므로, 대부분의 청크가 문서 중간에서 시작해 **BOS가 없다**.
Gemma는 BOS를 전제로 학습된 모델이라 이 입력이 사전학습 분포를 벗어난다.

| 모델 | BOS 있음 | BOS 없음 |
|---|---:|---:|
| Teacher 12B | 8.214 | **175.184** |
| Student Base 1B | 14.006 | 44.657 |

BOS가 없으면 수치가 부풀 뿐 아니라 **12B Teacher가 1B Student보다 4배 나쁘게 나온다.** 순서가 뒤집히므로
해석 불가능한 측정이다. 12B가 1B보다 BOS 부재에 훨씬 민감하다.

이 때문에 `prepend_bos` 옵션을 추가했고(청크마다 BOS를 붙이고 본문을 `seq_len - 1`로 자른다),
Gemma 설정에서는 **반드시 `prepend_bos: true`** 로 둔다. Qwen 설정은 기본값 `false`로 동작이 바뀌지 않는다.

KD를 BOS 없이 돌리면 PPL 175 상태의 Teacher logits을 목표 분포로 삼게 되므로, 증류 실험 자체가 무의미해진다.

## Teacher 양자화 — 한 장에 올리기

12B는 bf16 24.4GB로 3090 한 장(24GB)에 올라가지 않는다. 실측:

| 정밀도 | GPU | 가중치 | 피크(forward) | KL(T=2) vs bf16 | top-1 일치 |
|---|---|---:|---:|---:|---:|
| bf16 (2장 분할) | 0+1 | 24.37 GB | 26.69 GB | — | — |
| **int8** | 1장 | **13.24 GB** | 15.76 GB | **0.0849** | 82.1% |
| nf4 | 1장 | 7.81 GB | 11.87 GB | 0.1661 | 73.1% |

> ⚠️ 이 표는 pretrained Teacher, test 앞 16 chunk에서 쟀고, 기준 PPL이 126.9인 것으로 보아 BOS 없이 측정된 것으로 보인다.
> BOS를 붙인 전체 test에서 KD용 int8 Teacher (FT)를 다시 재면 KL(T=2) 0.012, top-1 일치 93.6%, PPL +0.6%다
> ([4-way 결과](gemma3-4way-topk-kd.md#kd에-실제로-쓴-teacher--int8--어댑터)).

`teacher_quantization: int8` 이면 Teacher가 한 장에 들어가므로, 이 저장소가 이미 검증한
rank별 전용 Teacher 배치를 그대로 쓸 수 있다:

```
rank 0: Student 1B → cuda:0 │ Teacher 12B int8 → cuda:2
rank 1: Student 1B → cuda:1 │ Teacher 12B int8 → cuda:3
```

별도 모델 병렬 코드 없이 4장을 쓴다. 다만 **양자화 오차는 KD 목표 분포에 그대로 실린다.**
int8이 nf4보다 KL 기준 2배 정확하므로 Teacher에는 int8을 권한다.

Teacher fine-tuning 단계는 학습이 필요하므로 `teacher_ft_quantization: nf4`(QLoRA)를 쓴다.
양자화된 base 위에서 LoRA만 학습한다.

## 측정하지 않은 것

1. **KD / FT Student.** Gemma용 학습 체크포인트가 없어 4-way 중 2개만 측정했다.
   나머지 둘은 Teacher FT → KD → baseline FT를 돌려야 나온다.
2. **Qwen 결과와의 직접 비교.** tokenizer가 다르다(Gemma 262,144 vs Qwen 151,936).
   PPL은 tokenization에 의존하므로 Qwen 실험의 PPL 11.68과 이 표의 8.21을 같은 축에서 비교하면 안 된다.
3. **속도의 공정성.** Teacher는 2장에 분할된 상태로, Student는 단일 GPU로 측정했다.
   또한 `evaluate_speed`는 `generate()`가 아니라 teacher-forcing forward 처리량이다.
4. **12B에서의 양자화-PPL 영향.** (후속 측정: FT Teacher 기준 int8 PPL 6.924 vs bf16 6.883, [4-way 결과](gemma3-4way-topk-kd.md#kd에-실제로-쓴-teacher--int8--어댑터)) 위 KL은 test 앞 16 chunk 기준이다. 전체 test split에서
   int8 Teacher의 PPL을 재지 않았다.
5. **KD 손실 버퍼.** Gemma vocab 262,208은 Qwen의 1.7배라 `kd_loss`의 fp32 logit 버퍼가
   batch 1 × seq 256 기준 스텝당 약 1.6–2.5GB다. batch를 올리면 여기가 먼저 막힌다.
   top-K logit 절단이 필요해지는 지점이다.

## 재현

```bash
# 평가 (BOS 정합, 권장)
PYTHONPATH=. .venv-gemma/bin/python main.py configs/gemma3_12b_1b_3090x4_eval.yaml --step evaluate
PYTHONPATH=. .venv-gemma/bin/python main.py configs/gemma3_12b_1b_3090x4_eval.yaml --step compare
```

Gemma 3는 gated 모델이다. `hf auth login`으로 토큰을 등록하고 두 repo의 라이선스에 동의해야 한다.
토큰은 `~/.cache/huggingface/token`에 두며 저장소에 넣지 않는다.

transformers는 Gemma 3 지원을 위해 4.50 이상이 필요하다(이 저장소의 Qwen 스택은 4.45.2).
별도 venv `.venv-gemma`를 쓰며 `requirements-gemma.txt`로 재현한다.
