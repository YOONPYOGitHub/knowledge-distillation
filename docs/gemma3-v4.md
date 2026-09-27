# Gemma 3 12B → 1B v4: 데이터 10배에서 KD가 FT를 크게 앞선다

측정일: 2026-09-27. 브랜치: `feature/gemma3-4way-topk-kd`.
환경: RTX 3090 24GB × 4, PyTorch 2.3.1+cu121, transformers 4.51.3, bitsandbytes 0.45.5.

[v3](gemma3-4way-topk-kd.md)는 한국어 위키 1,000문서로 학습했고, Student SFT가 epoch 1 이후 바로 과적합했다.
[v3 요인 분리](gemma3-v3-ablation.md)에서는 KD를 Base에서 시작해야 FT를 이긴다는 것을 확인했다(격차 중 4.3% 좁힘).
v4는 이 결론을 반영해 **데이터를 10배(10,000문서)로 늘리고, KD를 Base 초기값 + forward KL로** 돌렸다.
Qwen v4(`h100_qwen7b_korean_scale10k_v4`)와 같은 데이터 규모와 early stopping 설정이다.

## 결론

1. **KD가 FT를 크게, 확실하게 이긴다.** Student (KD) 7.866 vs Student (FT) 8.363으로 PPL이 **0.496 낮다(−5.9%)**.
   paired bootstrap 95% CI [−0.535, −0.459]이고, test chunk 832개 중 **724개(87%)**에서 KD가 낫다.
2. **Teacher와의 격차 중 KD가 좁힌 비율은 16.9%다.** 1,000문서 ablation의 같은 설정(Base + forward) 4.3%의 약 4배이고,
   Qwen v3의 3.1%보다 5배 크다. 데이터가 늘수록 KD가 전달하는 몫이 커진다.
3. **SFT는 데이터를 10배로 늘려도 여전히 epoch 1 이후 과적합한다.** 반면 KD는 3 epoch 동안 val CE가 계속 내려갔다.
   KD 항이 과적합을 막아 **같은 데이터를 더 오래 유효하게 학습**하게 해 준다는 v3 ablation의 해석이 더 큰 규모에서도 유지된다.

## 설정

v3 대비 바뀐 것만 적는다. 나머지(seq 256, BOS 부착, T=2, α=0.5, tokenmean, top-K 128, lr 1e-5, Adafactor, seed 42)는 v3와 같다.

| 항목 | v3 | v4 |
|---|---|---|
| 데이터 | 1,000문서 (train 1,327 chunk) | **10,000문서 (train 12,256 chunk)** |
| test split | 119 chunk | **832 chunk** (212,160 토큰) |
| KD 초기값 / divergence | SFT / reverse KL | **Base / forward KL** ([요인 분리](gemma3-v3-ablation.md) 결과) |
| epoch / early stopping | 8 / 없음 | **4 / patience 1, min_delta 0.005** (세 학습 단계 모두) |
| 재개 | 단계 단위 | **epoch 단위** (`resume_training: true`) |

Teacher FT, Student SFT, KD를 모두 새 데이터로 다시 학습했다. KD가 SFT에 의존하지 않으므로
Teacher FT(GPU 0, 1)와 Student SFT(GPU 2, 3)를 동시에 돌렸다(`scripts/run_gemma3_v4.sh`).

**test split이 v3와 다르므로 v3와 PPL 절대값을 비교하지 않는다.** 비교는 v4 안의 4-way와 bootstrap으로 하고,
실험 간 비교는 단위가 없는 "격차 중 KD가 좁힌 비율"로만 한다.

## 4-way 결과

test split 832 chunk × 255 토큰. Teacher는 bf16으로 GPU 2장 분할, Student는 cuda:0.

| 모델 | 파라미터 | PPL | tokens/s |
|---|---:|---:|---:|
| Teacher (FT, QLoRA) | 12,187,325,040 | **5.418** | 1,444 |
| **Student (KD)** | 999,885,952 | **7.866** | 4,559 |
| Student (FT) | 999,885,952 | 8.363 | 4,524 |
| Student (Base) | 999,885,952 | 12.253 | 4,523 |

- **Base → FT: 12.253 → 8.363 (−31.7%).** 1,000문서 v3의 −16.7%보다 SFT 이득도 크다.
- **FT → KD: 8.363 → 7.866 (−5.9%).**
- **격차 중 KD가 좁힌 비율 = (FT − KD) / (FT − Teacher) = 0.497 / 2.945 = 16.9%.**
- Student는 Teacher보다 약 3.2배 빠르고 파라미터는 1/12이다. tokens/s는 teacher-forcing forward 처리량이다(`generate()` 아님).
  Student 세 행의 차이는 같은 아키텍처의 측정 흔들림이다.

### paired bootstrap

같은 832개 test chunk에서 chunk 단위 NLL로 10,000회 복원추출했다(`scripts/gemma3_paired_bootstrap.py ... run`).

| A vs B | PPL(A) − PPL(B) | 95% CI | P(A가 나음) | A가 나은 chunk |
|---|---:|---:|---:|---:|
| **KD vs FT** | **−0.496** | **[−0.535, −0.459]** | 1.00 | **724 / 832** |
| KD vs Base | −4.386 | [−4.757, −4.028] | 1.00 | 735 / 832 |
| FT vs Base | −3.890 | [−4.245, −3.539] | 1.00 | 670 / 832 |

test가 v3보다 7배 커서 CI 폭도 좁아졌다(KD vs FT 폭 0.08, v3 ablation 약 0.24).

### 실험 간 비교: 격차 중 KD가 좁힌 비율

| 실험 | 데이터 | KD 설정 | KD vs FT (bootstrap) | 좁힌 비율 |
|---|---|---|---|---:|
| Qwen2.5 7B → 1.5B v3 | 1,000문서 | SFT 초기값 + reverse | 검정 안 함 | 3.1% |
| Gemma v3 | 1,000문서 | SFT 초기값 + reverse | 유의하지 않음 | 0.4% |
| Gemma v3 ablation | 1,000문서 | Base 초기값 + forward | 유의 | 4.3% |
| **Gemma v4** | **10,000문서** | **Base 초기값 + forward** | **유의 (CI 폭 0.08)** | **16.9%** |

Gemma v3 ablation과 v4는 KD 설정이 같고, 데이터 양과 그에 따른 Teacher·SFT 재학습, early stopping 설정만 다르다.
따라서 4.3% → 16.9%의 상승은 주로 **데이터 규모의 효과**로 본다. 다만 test split이 달라 엄밀한 통제 비교는 아니다.

## 학습 곡선

### Teacher FT (QLoRA nf4, GPU 0, 1, epoch당 약 170분)

| epoch | train CE | val CE |
|---:|---:|---:|
| 1 | 1.8398 | 1.8504 |
| 2 | 1.7645 | 1.8334 |
| 3 | 1.7178 | **1.8279** |

3 epoch 내내 개선됐다. 마지막 개선폭(0.0055)이 min_delta 0.005를 겨우 넘었으므로 epoch을 늘려도 이득은 작을 것이다.

### Student SFT (GPU 2, 3, epoch당 약 145분)

| epoch | train CE | val CE |
|---:|---:|---:|
| 1 | 2.2170 | **2.2428** |
| 2 | 1.7798 | 2.2942 |

**데이터를 10배로 늘려도 epoch 1 이후 과적합한다.** train CE는 0.44 떨어졌는데 val CE는 0.05 올랐다.
1B 전체 파인튜닝에서는 1만 문서도 한 번 보면 일반화 가능한 몫을 다 가져간다. epoch 2에서 early stopping.

### KD (Base 초기값 + forward KL, epoch당 약 185분)

| epoch | train CE | val CE | 저장 |
|---:|---:|---:|---|
| 1 | 2.1938 | 2.2065 | best |
| 2 | 1.8585 | **2.1761** | **best (최종)** |
| 3 | 1.6171 | 2.1721 | 개선폭 0.004 < min_delta, early stopping |

KD는 epoch 1부터 SFT best보다 낮고(2.2065 vs 2.2428), epoch 2에서 0.067 차이로 벌어진다.
같은 epoch 2에서 CE만 학습한 SFT는 2.2942로 나빠졌으므로, **KD의 이득은 더 오래 학습해서가 아니다.**

epoch 3의 val CE(2.1721)는 epoch 2보다 0.004 낮지만, 개선 판정과 best 저장이 같은 min_delta 규칙을 따르기 때문에
최종 체크포인트는 epoch 2다. 차이가 작아 결론에는 영향이 없다.

## 실행 기록 (KST)

| 시각 | 사건 |
|---|---|
| 9/27 03:33 | 순차 러너로 시작했다가 GPU 2, 3이 노는 것을 보고 3분 만에 중단 |
| 03:37 | 병렬 러너 첫 시도. GPU를 2장만 보이게 하자 KD용 `teacher_devices` 검증에 걸려 두 학습 모두 즉시 실패. `STOP_ON_FAIL`로 1분 안에 멈춤 |
| 03:38 | 수정 후 재시작. Teacher FT(GPU 0, 1) ‖ Student SFT(GPU 2, 3) |
| 08:27 | Student SFT 완료 (epoch 2에서 early stopping) |
| 12:09 | Teacher FT 완료 (3 epoch), KD 시작 |
| 15:14 / 18:18 | KD epoch 1 / 2 완료 |
| 18:20 | VESSL 워크스페이스 종료(18:29, 연장 불가)를 앞두고 epoch 3 시작 직후 수동 중단 |
| 18:28 | 워크스페이스 재시작 후 같은 명령으로 재개. **epoch 2까지 복원하고 epoch 3부터 이어감** |
| 21:35 | KD epoch 3, early stopping |
| 21:53 | 평가, 비교, bootstrap 완료 |

총 소요는 약 18시간 20분이다(중단 10분 포함). 병렬 러너가 SFT 약 4.8시간을 Teacher FT와 겹쳐서 줄였다.

재시작 과정에서 러너가 `distill.log`를 덮어써서 KD epoch 1~2의 진행 로그 본문은 사라졌다. epoch별 지표는 재개 파일의
history로 보존돼 `distill_history.json`에 모두 남아 있다. 러너는 이후 이어쓰도록 고쳤다.

## 한계

1. **seed가 하나다.** bootstrap은 test chunk 구성의 흔들림만 잡는다. 다만 KD vs FT의 CI가 0에서 멀어(상한 −0.459)
   seed를 바꿔도 결론이 뒤집힐 가능성은 낮다.
2. **요인이 완전히 분리되지는 않았다.** v3 ablation 대비 데이터 양, Teacher 재학습, early stopping 설정이 함께 바뀌었다.
3. **Qwen v4는 아직 결과가 없다.** 같은 규모에서 Qwen도 같은 경향인지는 모른다.
4. **Teacher 격차가 작아졌다.** FT와 Teacher의 PPL 차이가 v3 4.779에서 v4 2.945로 줄었다(test split이 달라 절대값 비교는 조심).
   격차 대비 비율로 보면 KD 몫은 커졌지만, 절대 개선폭으로 해석할 때는 이 점을 감안해야 한다.
5. **생성 품질은 보지 않았다.** PPL과 teacher-forcing 처리량만 쟀다.

## 산출물

| 파일 | 내용 |
|---|---|
| `results/gemma3/logs/gemma3-12b-1b-v4/evaluation_results.json` | 4-way PPL·속도 |
| `results/gemma3/logs/gemma3-12b-1b-v4/paired_bootstrap.json` | bootstrap 결과 |
| `results/gemma3/logs/gemma3-12b-1b-v4/{teacher,baseline,distill}_history.json` | 학습 곡선 |
| `results/gemma3/logs/gemma3-12b-1b-v4/summary.json`, `notes.md` | 자동 요약과 진단 |
| `results/gemma3/logs/gemma3-12b-1b-v4/*.log` | 단계별 실행 로그 (진행 표시줄 제외) |
| `results/gemma3/figures/gemma3-12b-1b-v4/` | PPL·속도·학습 곡선 차트 |

## 재현

```bash
# 약 18시간. 끊기면 같은 명령으로 다시 실행한다. 끝난 단계는 건너뛰고 끊긴 단계는 epoch 단위로 이어간다.
setsid nohup scripts/run_gemma3_v4.sh >> results/gemma3/logs/gemma3-12b-1b-v4/pipeline.log 2>&1 &
```

VESSL 워크스페이스는 연장이 안 되므로 24시간 안에 끝나지 않으면 재시작 후 같은 명령을 실행한다.
`/root`는 워크스페이스 볼륨이라 체크포인트와 재개 파일이 남는다.
