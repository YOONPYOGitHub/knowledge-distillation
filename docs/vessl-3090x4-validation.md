# VESSL 3090 × 4: 구현 및 검증 기록

검증일: 2026-09-20. 브랜치: `feature/vessl-3090x4-h100-parity`.

## 출발점과 보존

- Azure에서 확인한 마지막 실행 소스: `ed8790d` (v3). 학습 소스의 Git 변경 사항이 없었으며 주요 6개 파일의 SHA-256이 보존용 사본과 일치했다.
- 작업 사본은 `ae7fa1c`를 기반으로 VESSL용 변경을 적용했다. 원래 H100 브랜치를 변경하거나 Azure에 새 코드를 배포하지 않았다.
- v3 결과: KD PPL 11.68487, FT PPL 11.81362. v4 10k 설정은 존재하지만 조회한 Azure 위치에 완료 결과가 없었다.
- 소스·설정·기존 소형 로그/차트를 복사했다. macOS 가상환경·개인키·인증 설정·대형 체크포인트는 복사하지 않았다.

## 네 장을 사용하는 방식

| 프로세스 | 학습 Student | 고정 Teacher | 로컬 배치 |
|---|---|---|---|
| rank 0 | GPU 0 | GPU 2 | 1 |
| rank 1 | GPU 1 | GPU 3 | 1 |

Student만 2-way DDP로 동기화한다. Teacher는 같은 가중치의 추론용 복제본이다. 전역 배치는 H100 설정과 같은 2이고, 모델을 FSDP/텐서 병렬로 분할하는 방식은 아니다. **네 장의 메모리를 하나의 96GB 메모리처럼 사용하지 않는다.**

유지한 조건: 모델·Teacher/SFT 초기값 요구사항, 데이터 revision/문서 수/분할 seed, sequence 256, LR 1e-5, Adafactor, T=2, CE alpha=0.5, tokenmean Reverse KL, 8 epochs, early stopping 비활성화.

## 정확성을 위해 추가한 처리

- 명시적 rank별 Teacher 장치 매핑 및 Student와의 중복 배치 방지.
- `global_batch_size`를 지정한 경우 단일 Student와 여러 Student가 같은 전역 배치/순서를 사용한다.
- 마지막 홀수 샘플을 버리거나 중복 학습하지 않는다. 빈 rank는 label=-100인 dummy로 DDP에 참여하며 gradient 기여는 0이다.
- 로컬 loss를 `world_size × local_valid_tokens / global_valid_tokens`로 조정한다. DDP 평균 후 gradient는 전역 유효 토큰 평균이 된다. gradient clipping은 동기화 이후 수행한다.
- BF16 gradient의 rank 간 합산은 FP32로 수행한다. 로컬 BF16 forward/backward의 반올림까지 제거하는 것은 아니다.
- 검증 loss는 전역 배치 단위로 정규화한다. 전체 test 평가는 분산 sampler/collective에 들어가지 않는다.
- 분리 GPU 평가의 Teacher 입력 장치와 속도 측정 CUDA 동기화 장치를 수정했다.
- 학습 RNG와 데이터 분할 seed를 분리했다. 기존 설정은 `training_seed`를 생략하면 기존 RNG 초기화 동작을 유지한다.

## 실행한 검증

환경: RTX 3090 24GB × 4, Python 3.10.12, PyTorch 2.3.1+cu121, Transformers 4.45.2, datasets 3.6.0, PEFT 0.13.2, Accelerate 0.34.2.

| 검증 | 결과 |
|---|---|
| CPU 회귀 테스트 | **19개 통과** |
| 독립 4-rank DDP 통신 smoke | 데이터 분할·gradient 동기화 통과 |
| 2 Student + 2 Teacher 실제 GPU smoke | **통과**, 4장 모두 연산 수행 |
| 단일 전역 배치 대비 FP32 Forward/Reverse KL | gradient 최대 절대차 약 **4.47e-8**, 가중치 약 **5.6e-9** |
| BF16 Reverse KL | 실행·유한 loss·rank 간 동일 가중치 확인; 단일 배치 참조 대비 gradient 약 **9.77e-4**, 가중치 약 **3.05e-5** 차이 관측 |
| 홀수 마지막 배치·불균등 유효 토큰 | FP32 gradient 및 Adafactor 업데이트 비교 통과 |
| 실제 학습/검증 epoch 함수 | 작은 Qwen으로 통과 |
| 기존 호출부 호환성 | dataloader 5개, DDP wrapper 4개 호출 호환 확인 |

BF16 수치는 작은 랜덤 모델에서 관측한 값이며 일반 허용 오차나 실제 7B의 보증 범위가 아니다. 테스트 후 네 GPU 모두 학습 프로세스가 없음을 확인했다. 기본 이미지의 별도 Cairo 의존성을 보완한 뒤 `pip check`도 통과했다.

## 아직 검증하지 않은 것

1. **실제 7B Teacher + 1.5B Student 학습의 VRAM 최대치·속도·최종 PPL.** 소형 smoke 성공은 이를 대신하지 않는다.
2. **Azure 원본 체크포인트 전송과 실제 adapter 로딩.** 필수 Teacher adapter/Student FT는 원본 서버에서 확인했지만 VESSL로 전송하지 않았다. 네트워크 연결 경로와 PEFT 버전 호환성 확인이 필요하다.
3. **과거 H100 실행과 비트 동일 재현.** 과거 학습 RNG가 고정·저장되지 않았고, PyTorch/Transformers/CUDA 버전과 연산 장치도 다르다. 현재는 알고리즘·배치 동등성을 검증했으며 같은 PPL을 보장하지 않는다.
4. **처리량 향상.** 실제 모델에서 단일 3090 구성 대비 step time/유효 tokens per second를 측정해야 한다. 3090 네 장이 H100보다 빠르다는 의미가 아니다.
5. Windows 실기 접속은 이 세션에서 시험하지 않았다. Windows 매뉴얼은 PowerShell 구문과 접속 절차를 검토한 안내다.

엄밀한 후속 비교는 이 브랜치의 동일 sampler/학습 seed를 적용한 단일 H100 참조 실행(batch=2, global_batch=2, 별도 Teacher 매핑 없음)과 3090×4 실행(batch=1 × 2)을 동일 초기 가중치·고정 test split에서 비교한다. 과거 H100 PPL을 재현 불가능한 정확한 수치 목표로 두지 않는다.

실행 및 중단 조건은 [공통 가이드](vessl-3090x4-remote-explorer-guide.md), 접속은 [Windows](vessl-remote-explorer-windows.md) / [macOS](vessl-remote-explorer-macos.md)를 따른다.