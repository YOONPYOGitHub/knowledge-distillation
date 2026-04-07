# 상세 진행 일정

> 각 단계별 세부 작업 및 체크리스트

## 전체 로드맵

아래 파이프라인은 프로젝트의 전체 진행 흐름을 나타냅니다.  
데이터 준비 → Teacher 추론 → Student 학습(KD/FT) → 평가 & 비교 순서로 진행되며, 이 흐름이 아래 Phase 1~4의 세부 작업으로 구체화됩니다.

![Training Pipeline](diagrams/training-pipeline.svg)

---

## Phase 1: 프로젝트 준비

### 1.1 프로젝트 구조 및 문서 작성

| # | 세부 작업 | 상태 | 산출물 |
|---|----------|------|--------|
| 1.1.1 | 프로젝트 폴더 생성 | ✅ 완료 | `knowledge-distillation/` |
| 1.1.2 | README.md 작성 | ✅ 완료 | `README.md` |
| 1.1.3 | 모델 선택 가이드 작성 | ✅ 완료 | `docs/model-selection-guide.md` |
| 1.1.4 | 모델 설치 가이드 작성 | ✅ 완료 | `docs/model-installation-guide.md` |
| 1.1.5 | 실험 설계 다이어그램 작성 | ✅ 완료 | `docs/diagrams/*.svg` |
| 1.1.6 | 상세 일정표 작성 | ✅ 완료 | `docs/schedule.md` (이 문서) |
| 1.1.7 | 논문 레퍼런스 문서 작성 | ✅ 완료 | `docs/papers.md` |

### 1.2 개발 환경 구축

| # | 세부 작업 | 상태 | 확인 방법 |
|---|----------|------|----------|
| 1.2.1 | Python 3.10+ 설치 확인 | ✅ 완료 | `python --version` |
| 1.2.2 | uv로 가상환경 생성 | ✅ 완료 | `uv sync` |
| 1.2.3 | 디바이스 확인 (CUDA / MPS / CPU) | ✅ 완료 | 서버: `nvidia-smi` / 맥북: `torch.backends.mps.is_available()` |
| 1.2.4 | PyTorch 설치 (환경에 맞게) | ✅ 완료 | uv sync로 자동 설치 |
| 1.2.5 | pyproject.toml 의존성 설치 | ✅ 완료 | `uv sync` |
| 1.2.6 | 설치 검증 스크립트 실행 | ✅ 완료 | `uv run python verify_setup.py` (CUDA/MPS/CPU 자동 감지) |

### 1.3 환경 확인 및 모델 조합 결정

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 1.3.1 | GPU 서버 디바이스/VRAM 확인 | ⬜ 대기 | `nvidia-smi` (학교 서버) |
| 1.3.2 | 맥북 MPS 동작 확인 | ✅ 완료 | MPS 정상 동작 확인됨 |
| 1.3.3 | 모델 선택 가이드 참고하여 조합 결정 | ✅ 완료 | 로컬: gpt2→distilgpt2 / 서버: gpt2-large→gpt2 |
| 1.3.4 | Teacher 모델 다운로드 테스트 | ✅ 완료 | HF 캠시 정상 저장 |
| 1.3.5 | Student 모델 다운로드 테스트 | ✅ 완료 | distilgpt2 로드 확인 |
| 1.3.6 | 동시 로드 테스트 (서버) | ⬜ 대기 | Teacher + Student 동시에 GPU 적재 가능한지 |
| 1.3.7 | 맥북 소규모 테스트 | ✅ 완료 | gpt2 + distilgpt2 MPS에서 forward pass 확인 |

---

## Phase 2: 핵심 모듈 구현

### 2.1 설정 모듈 (config.py)

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 2.1.1 | 하이퍼파라미터 정의 | ✅ 완료 | KDConfig dataclass |
| 2.1.2 | 모델 경로 설정 | ✅ 완료 | teacher/student 모델명 |
| 2.1.3 | 데이터셋 설정 | ✅ 완료 | dataset name, seq_length 등 |
| 2.1.4 | 저장 경로 설정 | ✅ 완료 | run_id 기반 checkpoint/log/figure 경로 |
| 2.1.5 | YAML config 로딩 | ✅ 완료 | `from_yaml()` 함수, `configs/` 폴더 |

### 2.2 데이터 모듈 (dataset.py)

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 2.2.1 | WikiText-2 데이터셋 로드 | ✅ 완료 | `datasets` 라이브러리 사용 |
| 2.2.2 | 토크나이저 적용 | ✅ 완료 | max_length로 시퀀스 자르기 |
| 2.2.3 | Train/Val/Test 분리 | ✅ 완료 | 데이터셋에서 자동 분리 |
| 2.2.4 | DataLoader 생성 | ✅ 완료 | batch, shuffle, collate |
| 2.2.5 | 데이터 샘플 확인 | ✅ 완료 | 정상 토크나이징 확인 |

### 2.3 모델 모듈 (models.py)

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 2.3.1 | Teacher 모델 로드 함수 | ✅ 완료 | frozen, eval 모드 |
| 2.3.2 | Student 모델 로드 함수 | ✅ 완료 | trainable 모드 |
| 2.3.3 | 모델 정보 출력 유틸 | ✅ 완료 | 파라미터 수, 레이어 수 등 |
| 2.3.4 | FP16/양자화 옵션 지원 | ✅ 완료 | CUDA 서버용 FP16 옵션 |
| 2.3.5 | 로드 테스트 실행 | ✅ 완료 | 두 모델 정상 로드 확인 |

### 2.4 사전 검증

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 2.4.1 | Teacher forward pass 테스트 | ✅ 완료 | 입력 → logit 출력 확인 |
| 2.4.2 | Student forward pass 테스트 | ✅ 완료 | 입력 → logit 출력 확인 |
| 2.4.3 | logit 차원 일치 확인 | ✅ 완료 | vocab_size 동일한지 |
| 2.4.4 | 데이터 → 모델 흐름 테스트 | ✅ 완료 | 1 batch end-to-end |

---

## Phase 3: 학습 구현 및 실행

### 3.1 지식 증류 학습 (distill.py) — **핵심**

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 3.1.1 | KD Loss 함수 구현 | ✅ 완료 | KL Divergence + Temperature |
| 3.1.2 | Total Loss 함수 구현 | ✅ 완료 | α·CE + (1−α)·T²·KL |
| 3.1.3 | Training loop 구현 | ✅ 완료 | Teacher(frozen) → Student 학습 |
| 3.1.4 | Validation loop 구현 | ✅ 완료 | 매 epoch validation loss 추적 |
| 3.1.5 | Logging 구현 | ✅ 완료 | JSON 학습 로그 저장 |
| 3.1.6 | Checkpoint 저장 구현 | ✅ 완료 | best model 저장 |
| 3.1.7 | 소규모 데이터로 빠른 테스트 | ✅ 완료 | exp01: 1 epoch, seq=128 동작 확인 |
| 3.1.8 | 확장 실험 실행 | 🔄 진행 중 | exp02: 3 epochs, seq=256 실행 중 |

### 3.2 베이스라인 학습 (train_baseline.py)

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 3.2.1 | CE Loss 기반 학습 loop 구현 | ✅ 완료 | Hard label만 사용 |
| 3.2.2 | 동일 하이퍼파라미터 적용 | ✅ 완료 | 공정 비교를 위해 동일 조건 |
| 3.2.3 | Validation & Checkpoint | ✅ 완료 | 증류 학습과 동일 구조 |
| 3.2.4 | 확장 실험 실행 | 🔄 진행 중 | exp02: 3 epochs |

### 3.3 학습 모니터링

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 3.3.1 | Training loss 곡선 확인 | ✅ 완료 | training_loss.png 생성 |
| 3.3.2 | Validation loss 곡선 확인 | ✅ 완료 | history JSON 로그 포함 |
| 3.3.3 | 학습률 스케줄 확인 | ✅ 완료 | warmup + decay |
| 3.3.4 | GPU 메모리 사용량 추적 | ⬜ 대기 | 서버 실험 시 확인 |

---

## Phase 4: 평가 및 결과 분석

### 4.1 개별 모델 평가 (evaluate.py)

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 4.1.1 | Perplexity 계산 구현 | ✅ 완료 | exp(avg_loss) |
| 4.1.2 | 추론 속도 측정 구현 | ✅ 완료 | tokens/sec, latency |
| 4.1.3 | 메모리 사용량 측정 | ⬜ 대기 | 서버에서 peak VRAM 측정 예정 |
| 4.1.4 | 텍스트 생성 샘플 수집 | ⬜ 대기 | 동일 프롬프트로 생성 |
| 4.1.5 | Teacher 평가 실행 | ✅ 완료 | 기준선 (Upper Bound) |
| 4.1.6 | Student(KD) 평가 실행 | ✅ 완료 | 증류 효과 확인 |
| 4.1.7 | Student(FT) 평가 실행 | ✅ 완료 | Fine-tuning 기준선 |
| 4.1.8 | Student(Base) 평가 실행 | ✅ 완료 | 기준선 (Lower Bound, 원본) |

### 4.2 4-Way 비교 분석 (compare.py)

| # | 세부 작업 | 상태 | 설명 |
|---|----------|------|------|
| 4.2.1 | 결과 테이블 생성 | ✅ 완료 | 4개 모델 모든 지표를 테이블로 정리 |
| 4.2.2 | Perplexity 비교 차트 | ✅ 완료 | 막대 그래프 (4개 모델) |
| 4.2.3 | 학습 Loss 곡선 비교 | ✅ 완료 | KD vs FT 학습 곡선 |
| 4.2.4 | 추론 속도 비교 차트 | ✅ 완료 | tokens/sec 비교 |
| 4.2.5 | Q1: 압축 손실 분석 | ✅ 완료 | Teacher vs Student(KD) |
| 4.2.6 | Q2: 증류 효과 분석 | ✅ 완료 | Student(KD) vs Student(FT) ⭐ 핵심 |
| 4.2.7 | Q3: Fine-tuning 효과 분석 | ✅ 완룎 | Student(FT) vs Student(Base) |
| 4.2.8 | 텍스트 생성 정성 비교 | ⬜ 대기 | 같은 프롬프트, 4개 모델 출력 나란히 |
| 4.2.9 | 결과 자동 진단 + notes | ✅ 완료 | summary.json + notes.md 자동 생성 |

### 4.3 시각화 결과물

| 차트 | 유형 | 파일 |
|------|------|------|
| Training Loss 곡선 | Line Chart | `results/figures/{run_id}/training_loss.png` |
| Perplexity 비교 | Bar Chart | `results/figures/{run_id}/perplexity_comparison.png` |
| 추론 속도 비교 | Bar Chart | `results/figures/{run_id}/inference_speed.png` |

---

## Phase별 예상 소요 비중

```
Phase 1 (준비)    ████░░░░░░░░░░░░░░░░  20%
Phase 2 (구현)    ████████░░░░░░░░░░░░  35%
Phase 3 (학습)    ██████░░░░░░░░░░░░░░  25%
Phase 4 (분석)    ████░░░░░░░░░░░░░░░░  20%
```

## 체크포인트 & 마일스톤

| 마일스톤 | 완료 조건 | 상태 |
|---------|----------|------|
| **M1: 환경 준비 완료** | GPU 확인, 모델 다운로드, verify_setup.py 통과 | ✅ |
| **M2: 데이터 파이프라인 완료** | DataLoader에서 정상 배치 출력 확인 | ✅ |
| **M3: 1회 학습 성공** | distill.py로 1 epoch 완주, loss 감소 확인 | ✅ |
| **M4: 전체 학습 완료** | 증류 학습 + FT 학습 모두 완료 | ✅ |
| **M5: 4-Way 비교 완료** | 4개 모델 비교 차트 및 리포트 생성 | ✅ |

## 현재 진행 상황

- **현재 단계:** Phase 1~4 완료, exp02 (2차 실험) 진행 중
- **다음 단계:** GPU 서버에서 server_config로 본 실험 수행
