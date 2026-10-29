# 한국어 B급 LLM 비교 및 실버케어 적용 계획

## 목표

3명이 동일한 한국어 원문과 학습 조건에서 서로 다른 3B급 Teacher를 1B급 Student로 증류한다. 한국어 성능, 자원 효율, 안전성을 함께 비교해 실버케어 후속 모델 계열을 선정한다.

## 담당 모델

| 참가자 | Teacher | Student | 성격 | 접근 조건 |
|---|---|---|---|---|
| A | `Qwen/Qwen2.5-3B` | `Qwen/Qwen2.5-1.5B` | 현대 multilingual 계열 | 공개, Teacher 라이선스 검토 필요 |
| B | `EleutherAI/polyglot-ko-3.8b` | `EleutherAI/polyglot-ko-1.3b` | 한국어 특화 계열 | 공개, Apache-2.0 |
| C | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.2-1B` | 범용 baseline | Hugging Face 약관 승인 필요 |

각 Teacher/Student는 같은 family tokenizer를 사용해야 한다. 현재 방식은 vocabulary가 다르면 logit 차원의 의미가 달라져 KD를 수행할 수 없다.

## 공통 데이터

- Dataset: `wikimedia/wikipedia`
- Config: `20231101.ko`
- Revision: `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`
- License: CC BY-SA 3.0 / GFDL
- 규모: 한국어 문서 약 64.8만 개, 압축 약 783MB
- 필드: `id`, `url`, `title`, `text`
- Split: 원본 `train`을 seed 42로 고정 shuffle한 뒤 train/validation/test로 분리

영문 WikiText와 가장 유사한 한국어 Wikipedia 정제 corpus다. 첫 비교는 1,000개 문서로 smoke/run-time을 측정하고, 세 모델 모두 성공한 뒤 동일한 문서 수로 확대한다.

## 공정 비교 규칙

모델 이름 외에는 아래 값을 동일하게 유지한다.

| 항목 | 값 |
|---|---:|
| 원문 revision / shuffle seed | 고정 / 42 |
| 문서 수 | 1,000 |
| validation / test | 5% / 5% |
| sequence length | 128 |
| batch size | GPU당 1, global batch 2 |
| epochs | 1 |
| temperature / alpha | 2.0 / 0.3 |
| optimizer / learning rate | Adafactor / 2e-5 |
| GPU 방식 | 2-process DDP, 모델 복제 + 데이터 분할 |

Tokenizer가 다르면 같은 문장이 서로 다른 토큰 수가 되므로 모델 간 perplexity를 직접 비교하지 않는다. 원문 split은 같게 유지하고 tokenizer 효율 자체도 모델의 한국어 적합성으로 측정한다.

## 실행 순서

1. 각 VESSL에서 DDP smoke test를 통과시킨다.
2. 세 설정 모두 tokenizer vocabulary 호환성과 10~20 step smoke를 통과시킨다.
3. 추가 학습 전 Teacher와 Student(Base)를 같은 test split에서 평가한다.
4. `--step distill`로 KD를 실행한다. 첫 비교에서는 3B Teacher 전체 fine-tuning을 하지 않는다.
5. `--step baseline`으로 같은 Student를 CE-only 학습해 KD 효과를 분리한다.
6. Student(Base), Student(FT), Student(KD)를 공통 지표로 비교한다.
7. GPU peak memory, 학습 시간, 추론 tokens/sec도 함께 기록한다.

### VESSL 실행

두 GPU와 데이터 shard 동작을 작은 모델로 먼저 확인한다.

```bash
torchrun --standalone --nproc_per_node=2 scripts/ddp_smoke.py
```

각 참가자는 자신에게 배정된 설정 하나를 선택한다. 하나의 `torchrun`에서 KD, CE-only baseline, 평가, 비교를 순서대로 실행해야 같은 `run_id`와 checkpoint 경로를 사용한다.

```bash
torchrun --standalone --nproc_per_node=2 main.py \
	configs/exp09_korean_qwen.yaml \
	--step distill \
	--step baseline \
	--step evaluate \
	--step compare
```

다른 참가자는 설정 경로만 `exp10_korean_polyglot.yaml` 또는 `exp11_korean_llama.yaml`로 바꾼다.

DDP에서는 각 GPU가 모델 전체를 복제하고 서로 다른 데이터 shard를 처리한다. 모델은 GPU 사이에 분할하지 않는다.

## 평가 지표

### 한국어 언어 성능

- Bits-per-byte 또는 byte perplexity: tokenizer가 다른 모델 간 주 비교 지표
- Perplexity: 같은 family 내 Base/FT/KD 비교에만 사용
- Korean tokenizer fertility: 문자 또는 byte당 토큰 수
- KMMLU, KoBEST 등 한국어 이해 benchmark

### 실버케어 적합성

Wikipedia continual pretraining만으로 실버케어 모델이 완성되지는 않는다. 모델 family 선정 후 별도 domain SFT와 RAG 단계를 수행한다.

- 일상 대화: 존댓말, 쉬운 문장, 반복 질문 대응
- 안전 분류: 낙상, 흉통, 호흡곤란, 자해 위험의 즉시 escalation
- 의료 한계: 진단·처방 단정 금지, 의료진 연결 유도
- 복약 정보: 근거 문서 기반 답변과 출처 표시
- 개인정보: 실제 환자정보 없이 비식별 또는 합성 시나리오 사용
- 평가: 자동 점수만 사용하지 않고 노인 사용자/의료·돌봄 전문가의 blind review 수행

## 선정 원칙

최종 후보는 한국어 점수 하나로 정하지 않는다. 한국어 품질, 위험 응답 실패율, 추론 지연, VRAM, 라이선스, 배포 가능성을 함께 점수화한다. 실버케어에서는 일반 benchmark 상승보다 위험 상황 누락률과 근거 없는 의료 조언 비율을 우선한다.