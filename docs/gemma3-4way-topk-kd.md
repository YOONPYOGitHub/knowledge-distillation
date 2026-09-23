# Gemma 3 12B → 1B: 4-way KD 결과 (top-K vs full-vocab)

측정일: 2026-09-23. 브랜치: `feature/gemma3-4way-topk-kd`.
환경: RTX 3090 24GB × 4, PyTorch 2.3.1+cu121, transformers 4.51.3, bitsandbytes 0.45.5.

[증류 전 기준선 평가](gemma3-3090x4-eval.md)에 이어, Qwen v3(`h100_qwen7b_korean_reverse_v3`)와 **같은 KD 설정**으로
4-way(Teacher FT / Student KD / Student FT / Student Base)를 끝까지 돌린 결과다.
데이터 조건은 기준선 평가와 같다(한국어 위키 `20231101.ko`, revision 고정, 1,000문서, seq 256, BOS 부착).

## 결론

1. **Student는 SFT만으로 크게 좋아지고, KD는 그 위에 유의한 이득을 더하지 못했다.**
   Base 14.00 → FT 11.66 (−16.7%)은 확실하다. KD 11.64 vs FT 11.66의 차이 0.02는
   paired bootstrap 95% CI [−0.148, +0.118]이 0을 포함해 **노이즈와 구분되지 않는다.**
2. **top-K 128과 full-vocab KD의 결과도 구분되지 않는다.** top-K 11.643 vs full-vocab 11.699,
   CI [−0.156, +0.040]. top-K로 목표 분포를 잘라서 손해를 봤다는 증거는 없다.
3. **다만 top-K는 full-vocab의 "싼 근사"가 아니다.** 같은 Student 상태에서 gradient 코사인 0.883,
   상대 차이 46%로 학습 신호가 꽤 다르다. 최종 PPL이 같게 나온 것은 두 KD 모두 SFT 지점에서
   거의 움직이지 못했기 때문이지, 두 목적함수가 같아서가 아니다.
4. **top-K의 실익은 메모리뿐이었다.** KD 손실 버퍼 2.15 → 1.34GB(batch 1). 학습 시간은
   top-K 159분 / full-vocab 160분으로 같다. 이 구성(batch 1)에서는 full-vocab도 24GB 안에 들어간다.

자동 진단(`notes.md`, `summary.json`)은 KD < FT이면 무조건 "증류 효과 확인됨"으로 적는다.
이번 실행에서는 그 판정이 **틀렸다.** 판정 근거는 아래 bootstrap을 볼 것.


## 4-way 결과 (top-K 128)

test split 119 chunk × 256 토큰. Teacher는 bf16으로 GPU 2장 분할, Student는 cuda:0.

| 모델 | 파라미터 | PPL | tokens/s | ms/token |
|---|---:|---:|---:|---:|
| Teacher (FT, QLoRA) | 12,187,325,040 | **6.884** | 1,459 | 0.685 |
| Teacher (pretrained)¹ | 12,187,325,040 | 8.214 | — | — |
| Student (KD) | 999,885,952 | **11.643** | 4,594 | 0.218 |
| Student (FT) | 999,885,952 | **11.663** | 4,521 | 0.221 |
| Student (Base) | 999,885,952 | **14.004** | 4,535 | 0.221 |

¹ [기준선 평가](gemma3-3090x4-eval.md)의 값. 이번 실행은 `attn_implementation: eager`라 조건이 한 가지 다르지만,
seq 256에서 eager의 PPL 영향은 +0.005 수준이다. 실제로 같은 Student (Base)가 두 실행에서 14.006 / 14.004로 일치한다.

- **Teacher FT는 확실히 먹혔다.** 8.214 → 6.884 (−16.2%).
- **Student는 Base → FT에서 크게 좋아진다.** 14.004 → 11.663 (−16.7%).
- **KD가 FT 위에 얹은 몫은 0.020 (0.17%)이고, 노이즈와 구분되지 않는다.** [해석](#해석-시-주의)을 볼 것.

tokens/s는 **이 표 안에서만** 비교한다. 기준선 평가의 Student (Base) 7,214 tokens/s와 다른 것은
평가 batch(2 → 1)와 attention 구현(기본값 → eager)이 달라서다. 모델이 느려진 것이 아니다.

## 학습 곡선

### Teacher FT (QLoRA nf4, 3 epoch, epoch당 약 18분)

| epoch | train CE | val CE |
|---:|---:|---:|
| 1 | 2.0153 | 1.7118 |
| 2 | 1.8897 | 1.6881 |
| 3 | 1.8293 | **1.6826** |

단조 개선. 3 epoch에서도 아직 내려가고 있어 과적합 신호가 없다.

### Student SFT (baseline, 8 epoch, epoch당 약 16분)

| epoch | train CE | val CE |
|---:|---:|---:|
| 1 | 2.3676 | **2.0850** |
| 2 | 1.7413 | 2.1779 |
| 3 | 1.2173 | 2.4740 |
| 4 | 0.8081 | 2.9753 |
| 5 | 0.5275 | 3.4892 |
| 6 | 0.3547 | 3.9218 |
| 7 | 0.2650 | 4.1752 |
| 8 | 0.2139 | 4.3509 |

**epoch 1 이후 심한 과적합.** train CE는 0.21까지 떨어지는데 val CE는 2.09 → 4.35로 두 배가 된다.
1,000문서(train 1,327 chunk)에 1B 모델을 8 epoch 돌리면 학습 데이터를 외운다.
best만 저장하므로 `student_ft_best.pt`는 **사실상 epoch 1 체크포인트**이고, 이것이 Student (FT) 행이자 KD의 초기값이다.

### KD (top-K 128 reverse KL, 8 epoch, epoch당 약 20분)

| epoch | train loss | val CE |
|---:|---:|---:|
| 1 | 1.3445 | 2.0788 |
| 2 | 1.1480 | **2.0736** |
| 3 | 1.0078 | 2.0900 |
| 4 | 0.9011 | 2.1124 |
| 5 | 0.8152 | 2.1144 |
| 6 | 0.7433 | 2.1424 |
| 7 | 0.6865 | 2.1438 |
| 8 | 0.6418 | 2.1541 |

best는 epoch 2. 이후 train loss는 계속 내려가지만 val CE는 단조 악화다.
SFT보다 악화 속도가 훨씬 느린 것(8 epoch 뒤 +0.08 vs SFT +2.27)은 KD 항이 정규화로 작용하기 때문으로 보이나,
best 지점의 val CE 자체는 SFT best(2.0850)보다 0.011 낮은 데 그친다.

best 선택 기준은 `val_ce_loss`다. `Val Loss`(CE+KD 합)가 아니다. epoch 2는 합계로는 epoch 1보다 나쁘지만(1.5360 vs 1.5344) CE로는 더 좋다.

## top-K vs full-vocab A/B

두 실행은 Teacher FT 어댑터와 KD 초기값(`student_ft_best.pt`)을 공유하고 `kd_top_k`(128 vs 0)만 다르다.
따라서 Student (KD) 행의 차이가 곧 목표 분포를 자른 대가다.

| | top-K 128 | full-vocab | 차이 |
|---|---:|---:|---:|
| Student (KD) PPL | **11.643** | 11.699 | +0.056 |
| FT 대비 | −0.020 (−0.17%) | +0.036 (+0.31%) | |
| best epoch (val CE) | 2 (2.0736) | 1 (2.0960) | |
| 8 epoch 후 val CE | 2.1541 | 2.2139 | |
| KD 손실 버퍼 (batch 1) | 1.34 GB | 2.15 GB | −38% |
| KD 학습 시간 (8 epoch) | 159분 | 160분 | 같음 |

full-vocab은 epoch 1부터 val CE가 나빠지기 시작해 top-K보다 과적합이 빠르고 깊다.

### 학습 곡선 비교 (val CE)

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| top-K 128 | 2.0788 | **2.0736** | 2.0900 | 2.1124 | 2.1144 | 2.1424 | 2.1438 | 2.1541 |
| full-vocab | **2.0960** | 2.1035 | 2.1292 | 2.1507 | 2.1566 | 2.1794 | 2.1901 | 2.2139 |

`Val Loss`(CE+KD 합)는 두 실행 사이에서 **비교하지 않는다.** top-K는 상위 128개 안에서 다시 정규화한 분포의
KL이라 KD 항의 크기 자체가 다르다(같은 배치에서 full 2.02 vs top-K 1.10). CE만 같은 축이다.

### 정적 측정 — 같은 Student 상태, 16개 배치

학습 없이, 같은 Teacher/Student/배치에서 두 방식의 KD 손실과 Student gradient를 직접 비교했다
(`scripts/gemma3_topk_ablation.py`, 결과 `results/gemma3/logs/gemma3-12b-1b-v3/topk_ablation.json`).

| 지표 | 값 |
|---|---:|
| KD 손실 | full 2.0188 → top-K 1.1021 |
| CE 손실 | 2.2884 (두 방식 동일 — 정상) |
| gradient 코사인 | **0.883** (배치별 최저 0.686) |
| gradient norm 비 (top-K / full) | 0.927 |
| \|Δg\| / \|g\| | **0.463** |

코사인 0.88은 방향이 대체로 같다는 뜻이지만, 상대 차이 46%는 "같은 방향으로 더 싸게 가는 근사"라고 부르기엔 크다.
top-K KD는 full-vocab KD와 **다른 목적함수**로 보는 것이 맞다. Qwen v3 KD와 알고리즘이 같다고 말하려면
full-vocab 쪽 수치를 써야 한다.

## 해석 시 주의

### PPL 차이가 노이즈를 넘는가 — paired bootstrap

evaluate는 test 전체 PPL 한 값만 내므로 0.02 차이가 의미 있는지 판단할 수 없다.
`scripts/gemma3_paired_bootstrap.py`로 같은 119개 test chunk에서 모델별 chunk 단위 NLL을 구하고,
chunk를 10,000번 복원추출해 PPL 차이의 분포를 만들었다. 두 모델이 같은 chunk를 보므로 chunk 난이도 차이는 상쇄된다.

| A vs B | PPL(A) − PPL(B) | 95% CI | P(A가 나음) | A가 나은 chunk |
|---|---:|---:|---:|---:|
| KD top-K vs FT | −0.012 | [−0.148, +0.118] | 0.55 | 49 / 119 |
| KD full-vocab vs FT | +0.044 | [−0.057, +0.144] | 0.20 | 54 / 119 |
| KD top-K vs KD full-vocab | −0.055 | [−0.156, +0.040] | 0.87 | 58 / 119 |
| FT vs Base | **−2.348** | **[−3.145, −1.615]** | 1.00 | 76 / 119 |

**CI가 0을 포함하지 않는 것은 FT vs Base 하나뿐이다.** KD top-K는 평균 PPL이 FT보다 낮지만
chunk 단위로는 119개 중 49개에서만 FT보다 낫다.

bootstrap 스크립트는 logits를 fp32로 올려 CE를 계산하고, evaluate는 bf16 logits 그대로 계산한다.
그래서 PPL이 조금 다르다(FT 11.654 vs 11.663, KD top-K 11.642 vs 11.643, Base 14.002 vs 14.004).
**수치 정밀도 차이만으로 FT가 0.009 움직인다** — KD−FT 차이 0.020의 절반에 가까운 크기다.

### Student SFT의 과적합

`student_ft_best.pt`는 epoch 1 체크포인트다(그 뒤로 val CE가 두 배까지 나빠진다). 즉 이 실험의 Student (FT)는
"1 epoch SFT"이고, KD는 거기서 출발해 1–2 epoch 안에 best를 찍고 나빠진다.
KD가 SFT 지점에서 거의 움직이지 못한 것은 **데이터 1,000문서가 KD 신호를 받아들이기에 너무 작다**는 쪽이
더 그럴듯한 설명이다. KD 알고리즘(top-K / full-vocab)을 바꿔도 같은 결과가 나온 것이 이를 뒷받침한다.

### Qwen v3와의 상대 비교

PPL은 tokenizer에 의존하므로 절대값은 비교하지 않는다. 같은 모델 계열 안의 **상대 변화**만 본다.

| | Qwen2.5 7B → 1.5B v3 | Gemma 3 12B → 1B v3 |
|---|---:|---:|
| Base → FT | 12.127 → 11.814 (−2.6%) | 14.004 → 11.663 (−16.7%) |
| FT → KD | 11.814 → 11.685 (**−1.09%**) | 11.663 → 11.643 (**−0.17%**) |

Qwen v3에서 KD가 FT 대비 1.1% 이득을 냈다면, Gemma에서는 그 1/6 수준이다. Gemma 1B pt는 한국어 위키에서
SFT로 얻는 이득이 훨씬 커서(−16.7% vs −2.6%) KD가 개선할 여지가 SFT 단계에서 이미 소진된 것으로 보인다.
다만 **Qwen v3의 1.1%도 bootstrap으로 검정하지 않았다.** 같은 판정을 Qwen 쪽에도 적용해야 두 결과를 나란히 놓을 수 있다.

### 속도

tokens/s는 teacher-forcing forward 처리량이다(`generate()` 아님). Student 세 행(4,521–4,594)의 차이는
같은 아키텍처의 측정 흔들림이다. KD가 모델을 빠르게 만든 것이 아니다.

## 실행 기록

- **Teacher FT → Student SFT**: 2026-09-22 17:47 – 20:51, 중단 없이 완주.
- **KD 1차**: 09-22 20:51 시작, epoch 3까지 완료하고 epoch 4의 62%(412/664 step)에서 torchrun이
  외부 SIGTERM(signal 15)을 받아 종료. OOM killer는 SIGKILL을 보내므로 원인이 아니고, 디스크·RAM 여유도 충분했다.
  원격 세션 종료가 torchrun에 전달된 것으로 본다. 재실행 중 세션이 다시 끊겼을 때 `setsid`로 분리한 프로세스는
  살아남았다는 점이 이를 뒷받침한다.
- **KD 재실행**: 앞 두 단계의 산출물을 재사용해 KD부터 다시 시작. 세션과 분리(`setsid nohup`)해서 실행했다.
  seed 42 고정이라 **epoch 1–3의 val loss가 1차 실행과 소수점 4자리까지 일치**했다(1.5344 / 1.5360 / 1.5542).
  09-23 12:07 시작, 14:46 KD 완료, 14:48 평가·비교까지 완료.
- **full-vocab A/B**: 본 실행의 `PIPELINE_DONE`을 기다렸다가 14:48 자동 시작, 17:33 완료
  (KD 160분 → 평가 1분 → 정적 측정 2분 → 요약표).
- **paired bootstrap**: A/B 종료 후 GPU가 빈 상태에서 별도 실행(약 3분).

## 측정하지 않은 것

1. **다른 seed.** 모든 실행이 seed 42 하나다. bootstrap은 test chunk 구성의 흔들림만 잡고,
   학습 seed에 따른 흔들림은 잡지 못한다. KD vs FT가 seed를 바꿔도 같은 결론인지는 모른다.
2. **더 큰 데이터.** 1,000문서가 KD 이득을 막고 있다는 해석은 가설이다. 문서 수를 늘린 실행이 필요하다.
3. **early stopping을 켠 KD.** `early_stopping_patience: 0`(Qwen v3 parity)이라 8 epoch을 다 돌았다.
   best만 저장하므로 PPL 결과는 같겠지만, 학습 시간은 top-K 기준 약 2/3을 줄일 수 있었다.
4. **다른 K.** K = 128 하나만 봤다. gradient 상대 차이 46%가 K에 따라 어떻게 줄어드는지 모른다.
5. **생성 품질.** PPL과 처리량만 쟀다. 한국어 생성 결과의 정성 비교는 하지 않았다.

## 산출물

| 파일 | 내용 |
|---|---|
| `results/gemma3/logs/gemma3-12b-1b-v3/evaluation_results.json` | 4-way PPL·속도 (top-K) |
| `results/gemma3/logs/gemma3-12b-1b-v3-fullvocab/evaluation_results.json` | 4-way PPL·속도 (full-vocab) |
| `results/gemma3/logs/gemma3-12b-1b-v3/{teacher,baseline,distill}_history.json` | 학습 곡선 |
| `results/gemma3/logs/gemma3-12b-1b-v3-fullvocab/distill_history.json` | full-vocab KD 학습 곡선 |
| `results/gemma3/logs/gemma3-12b-1b-v3/topk_ablation.json` | 정적 gradient 비교 (배치별) |
| `results/gemma3/logs/gemma3-12b-1b-v3/paired_bootstrap.json` | bootstrap 결과 |
| `results/gemma3/figures/gemma3-12b-1b-v3{,-fullvocab}/` | PPL·속도·학습 곡선 차트 |
| `results/gemma3/logs/gemma3-12b-1b-v3/pipeline.log`, `…-fullvocab/ab.log` | 전체 실행 로그 (1차 중단분 포함) |

체크포인트(`results/gemma3/checkpoints/`, Student 2GB × 3 + LoRA 어댑터)는 저장소에 넣지 않는다.

## 재현

```bash
# 처음부터 (Teacher FT → Student SFT → KD → 평가, 약 6시간)
scripts/run_gemma3_4way.sh configs/gemma3_12b_1b_3090x4_v3.yaml \
  > results/gemma3/logs/gemma3-12b-1b-v3/pipeline.log 2>&1

# 중단된 실행 이어가기 — 앞 단계 체크포인트가 남아 있으면 해당 단계부터
scripts/run_gemma3_4way.sh configs/gemma3_12b_1b_3090x4_v3.yaml distill

# full-vocab 대조군 (본 실행의 PIPELINE_DONE을 기다렸다가 시작)
scripts/run_gemma3_fullvocab_ab.sh

# KD / FT 차이가 노이즈를 넘는지 판정 (두 실행이 끝난 뒤, GPU 1장)
PYTHONPATH=. .venv-gemma/bin/python scripts/gemma3_paired_bootstrap.py
```

장시간 실행은 `setsid nohup ... &`로 세션에서 분리한다. 원격 접속이 끊기면 torchrun이 SIGTERM을 받아 학습이 죽는다.

시작 단계는 `train_teacher | baseline | distill | evaluate | compare` 중 하나다.
`distill`부터 시작하려면 `teacher_checkpoint`(LoRA 어댑터)와 `distill_student_checkpoint`(SFT)가 있어야 한다.
없으면 `scripts/gemma3_stage.py`가 pretrained 모델로 조건을 낮춰 진행하고 로그에 ⚠️를 남긴다.
