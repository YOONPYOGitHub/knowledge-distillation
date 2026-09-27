# Gemma 3 12B → 1B: v3 KD 요인 분리 (초기값 × divergence)

측정일: 2026-09-26. 브랜치: `feature/gemma3-4way-topk-kd`.
환경: RTX 3090 24GB × 4, PyTorch 2.3.1+cu121, transformers 4.51.3, bitsandbytes 0.45.5.

[v3 4-way 결과](gemma3-4way-topk-kd.md)에서 KD는 FT와 노이즈 안에서 구분되지 않았다(11.643 vs 11.663).
v3 설정은 Qwen v3를 그대로 따른 것인데, Qwen v3는 이전 실험 대비 **KD 초기값(Base → SFT)과
divergence(forward → reverse KL)를 동시에 바꿨다.** 어느 쪽이 효과를 냈는지, 또는 어느 쪽이 방해했는지를
2×2로 나눠서 확인했다.

## 결론

1. **KD 초기값이 결과를 가른다. Base에서 시작한 KD는 FT를 유의하게 이긴다.**
   Base+forward KD 11.457은 FT 11.663보다 0.194 낮고(95% CI [−0.320, −0.080]),
   Base+reverse KD 11.507도 0.142 낮다(CI [−0.263, −0.027]). Gemma에서 **처음으로 KD > FT가 통계적으로 확인됐다.**
2. **SFT에서 시작하면 KD 이득이 사라진다.** FT 초기값의 두 KD는 divergence와 무관하게 FT와 구분되지 않는다
   (reverse −0.012, forward +0.009, 둘 다 CI가 0을 포함). 같은 divergence에서 Base 초기값이 FT 초기값보다
   유의하게 낫다(reverse −0.130, forward −0.203, 둘 다 CI가 0을 포함하지 않음).
3. **divergence 방향은 차이를 만들지 못했다.** 같은 초기값에서 reverse와 forward의 차이는 CI가 0을 포함한다.
   Base 초기값에서는 forward가 0.053 낮지만(P = 0.94) 유의하지 않다.
4. **따라서 Qwen v3의 두 변경 중 SFT 초기화는 Gemma에서는 오히려 KD 효과를 없앴다.**
   Qwen v3에서 KD가 FT를 이긴 원인이 무엇이었는지는 이 결과로 알 수 없다(모델이 다르고, Qwen 쪽은 bootstrap을 하지 않았다).

## 설계

나머지 조건은 모두 v3와 같다: 한국어 위키 1,000문서, seq 256, BOS 부착, T=2, α=0.5, lr 1e-5, tokenmean,
top-K 128, Adafactor, seed 42. Teacher FT 어댑터와 Student SFT 체크포인트는 v3 실행의 것을 그대로 쓴다.
따라서 네 실행의 Teacher (FT) 6.884 / Student (FT) 11.663 / Student (Base) 14.004가 모두 같다.

|                 | reverse KL | forward KL |
|---|---|---|
| **FT 초기값**   | v3 (기존) | `abl_ft_fwd` |
| **Base 초기값** | `abl_base_rev` | `abl_base_fwd` |

새로 돌린 세 실행은 early stopping(patience 2, min_delta 0)을 켰다. best 선택 규칙(val CE 최저)은 v3와 같고,
v3의 기존 곡선에 적용해도 같은 epoch 2를 고른다. 세 실행 모두 epoch 2가 best였고 epoch 4에서 멈췄다.

## 결과

### Student (KD) PPL

|                 | reverse KL | forward KL |
|---|---:|---:|
| **FT 초기값**   | 11.643 | 11.660 |
| **Base 초기값** | 11.507 | **11.457** |

기준: Student (FT) 11.663, Student (Base) 14.004, Teacher (FT) 6.884.

| 실행 | KD PPL | FT 대비 | 격차 중 KD가 좁힌 비율 | best epoch | best val CE | KD 학습 |
|---|---:|---:|---:|---:|---:|---:|
| FT + reverse (v3) | 11.643 | −0.020 | 0.4% | 2 | 2.0736 | 158분 (8 epoch) |
| FT + forward | 11.660 | −0.003 | 0.1% | 2 | 2.0662 | 80분 (4 epoch) |
| Base + reverse | 11.507 | −0.156 | 3.3% | 2 | 2.0677 | 80분 (4 epoch) |
| **Base + forward** | **11.457** | **−0.206** | **4.3%** | 2 | **2.0565** | 80분 (4 epoch) |

격차 중 KD가 좁힌 비율 = (FT − KD) / (FT − Teacher). [v3 문서](gemma3-4way-topk-kd.md#qwen-v3와의-상대-비교)의
정의와 같다. Qwen v3는 3.1%였으므로, Base 초기값 KD는 **Qwen v3와 같거나 조금 큰 수준**이다.

### paired bootstrap

같은 119개 test chunk에서 chunk 단위 NLL로 10,000회 복원추출했다(`scripts/gemma3_paired_bootstrap.py ... ablation`).
PPL(A) − PPL(B)가 음수면 A가 낫다. bootstrap의 관측값은 chunk 단위 계산이라 평가 단계의 PPL 차이와 소수점 셋째 자리에서 다를 수 있다.

| 비교 | A − B | 95% CI | P(A가 나음) | 유의 |
|---|---:|---:|---:|---|
| **Base 초기값 KD vs FT** | | | | |
| Base+forward vs FT | **−0.194** | [−0.320, −0.080] | 1.00 | ✅ |
| Base+reverse vs FT | **−0.142** | [−0.263, −0.027] | 0.99 | ✅ |
| **FT 초기값 KD vs FT** | | | | |
| FT+reverse (v3) vs FT | −0.012 | [−0.151, +0.120] | 0.56 | — |
| FT+forward vs FT | +0.009 | [−0.142, +0.151] | 0.45 | — |
| **초기값 효과 (divergence 고정)** | | | | |
| FT+reverse vs Base+reverse | +0.130 | [+0.033, +0.224] | 0.00 | ✅ Base가 나음 |
| FT+forward vs Base+forward | +0.203 | [+0.128, +0.278] | 0.00 | ✅ Base가 나음 |
| **divergence 효과 (초기값 고정)** | | | | |
| FT+reverse vs FT+forward | −0.020 | [−0.074, +0.033] | 0.78 | — |
| Base+reverse vs Base+forward | +0.053 | [−0.014, +0.123] | 0.06 | — |

각 실행의 자동 진단(`notes.md`)은 KD < FT이면 무조건 "증류 효과 확인됨"으로 적는다. FT+forward(11.660 vs 11.663)도
그렇게 적혀 있지만 **이 판정은 틀렸다.** 판정은 위 표를 따른다.

### 학습 곡선 (val CE)

| epoch | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| FT + reverse (v3) | 2.0788 | **2.0736** | 2.0900 | 2.1124 |
| FT + forward | 2.0706 | **2.0662** | 2.0934 | 2.1133 |
| Base + reverse | 2.0946 | **2.0677** | 2.0681 | 2.0839 |
| Base + forward | 2.0799 | **2.0565** | 2.0685 | 2.0814 |
| (참고) Student SFT, Base에서 CE만 | **2.0850** | 2.1779 | 2.4740 | 2.9753 |

v3는 8 epoch을 돌았지만 비교를 위해 4 epoch까지만 적었다.

## 해석

### KD의 이득은 "더 오래 학습해서"가 아니다

Base 초기값 KD는 best까지 2 epoch을 학습했고, SFT의 best는 1 epoch이다. 학습량 차이로 설명할 수 있는지 확인하려면
Base에서 CE만으로 2 epoch 학습한 결과를 보면 된다. 그것이 SFT epoch 2이고, val CE는 **2.178로 오히려 나빠졌다.**
같은 2 epoch에서 KD는 2.057로 SFT best(2.085)보다 낮다. 데이터 1,000문서에서 CE만으로는 epoch 1 이후 외우기 시작하는데,
KD 항이 이를 막으면서 epoch 2까지 일반화를 계속 끌어올린 것이다.

### SFT에서 시작하면 왜 이득이 사라지나

가설이다. SFT epoch 1 체크포인트는 이미 이 데이터의 일반화 가능한 부분을 CE로 흡수한 상태이고,
다음 epoch부터는 외우는 방향으로 가기 직전이다. 여기서 KD를 시작하면 KD 항이 과적합을 늦추는 역할만 하고
(v3 8 epoch 뒤 val CE +0.08 vs SFT +2.27), 새 정보를 전달할 여지는 거의 없다.
Base에서 시작하면 Student가 이 데이터에서 배우는 **첫 epoch부터 Teacher 분포를 함께 따라가므로**,
정답 토큰 외의 확률 정보가 일반화에 반영된다.

MiniLLM이 SFT 초기화를 쓰는 것은 instruction 데이터에서 on-policy 생성 학습을 하기 위해서다.
이 실험은 plain text에서 off-policy token-level KD이므로 그 이유가 적용되지 않는다.

## 한계

1. **seed가 하나다.** bootstrap은 test chunk 구성의 흔들림만 잡는다. 특히 divergence 비교(P = 0.06)는 seed를 바꾸면 뒤집힐 수 있다.
2. **데이터가 1,000문서다.** 데이터가 많아져 SFT가 여러 epoch 동안 개선되는 조건에서도 Base 초기값이 나은지는 모른다.
   v4(10,000문서)에서 확인한다.
3. **α, lr, T는 고정했다.** 2×2는 Qwen v3가 바꾼 요인 중 두 개만 분리한다.
4. **Qwen에서는 확인하지 않았다.** 같은 2×2를 Qwen에 적용해야 모델 계열에 무관한 결론인지 알 수 있다.

## v4에 반영

v4의 KD는 **Base 초기값 + forward KL**로 한다.

- 초기값: Base가 두 divergence 모두에서 유의하게 나았다.
- divergence: 유의한 차이는 없었다. 관측값이 가장 좋았던 forward를 쓴다. 고전적 KD 설정이기도 하다.
- Student SFT는 KD 초기값으로는 쓰지 않지만 4-way의 Student (FT) 행을 위해 그대로 학습한다.

## 산출물

| 파일 | 내용 |
|---|---|
| `results/gemma3/logs/gemma3-12b-1b-v3/ablation_summary.json` | 2×2 요약 (PPL, best epoch, val CE 곡선) |
| `results/gemma3/logs/gemma3-12b-1b-v3/paired_bootstrap_ablation.json` | bootstrap 결과 |
| `results/gemma3/logs/gemma3-12b-1b-v3-abl-{ft-fwd,base-rev,base-fwd}/` | 실행별 평가 결과, KD 학습 곡선, 자동 진단 |
| `results/gemma3/figures/gemma3-12b-1b-v3-abl-*/` | 실행별 PPL·속도 차트 |
| `results/gemma3/logs/gemma3-12b-1b-v3-ablation/ablation.log` | 전체 실행 로그 (진행 표시줄 제외) |

## 재현

```bash
# v3 실행(Teacher FT 어댑터, student_ft_best.pt)이 먼저 있어야 한다. 약 4시간.
setsid nohup scripts/run_gemma3_ablation.sh \
  > results/gemma3/logs/gemma3-12b-1b-v3-ablation/ablation.log 2>&1 &

# 중단되면 남은 변형부터 (ft_fwd | base_rev | base_fwd)
scripts/run_gemma3_ablation.sh base_rev
```

실행 기록: 2026-09-26 13:55 시작, 18:07 완료. 세 실행 모두 중단 없이 완주했다(KD 80 / 80 / 80분, 평가 각 1분,
bootstrap 3분). 실행 중 원격 접속이 한 번 끊겼지만 `setsid`로 분리해 둔 실행은 영향을 받지 않았다.
