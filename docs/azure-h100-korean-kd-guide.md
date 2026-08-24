# Azure H100 한국어 지식 증류 가이드

대상 VM은 단일 `NVIDIA H100 NVL 94GB`이므로 DDP를 사용하지 않는다. `python main.py ...`로 실행하면 기존 코드는 자동으로 단일 프로세스 경로를 사용한다.

## 저장 경로

- 소스: `/var/lib/knowledge-distillation`
- Python 환경: `/var/lib/kd-venv`
- Hugging Face 모델/데이터 캐시: `/var/lib/huggingface`
- 결과: `/var/lib/kd-results/<model>`

OS 디스크를 128GB로 확장했으며 `/var/lib`에 소스, 환경, 모델 캐시와 결과를
보존한다. `/mnt`는 Spot eviction 또는 호스트 이동 시 초기화되므로 학습 상태나
필수 캐시 저장에 사용하지 않는다. 완료된 JSON과 차트는 추가로 로컬에 회수한다.

## 환경 설치

```bash
bash scripts/setup_azure_h100.sh
```

## Qwen smoke

```bash
bash scripts/run_azure_h100.sh configs/h100_qwen_korean_smoke.yaml --step distill
```

## 프로젝트 목적 4-Way 실험

마지막 로컬 exp08과 같은 비교 구조를 실행한다. Teacher를 한국어 데이터에 먼저 fine-tuning하고, 같은 Student와 동일 학습 조건으로 KD와 CE-only baseline을 각각 8 epoch 학습한 뒤 Teacher/KD/FT/Base를 평가한다.

```bash
scripts/start_azure_h100_job.sh configs/h100_qwen_korean_4way.yaml
scripts/monitor_azure_h100.sh --watch
```

`--step`을 전달하지 않아야 `train_teacher`, `distill`, `baseline`, `evaluate`, `compare`가 같은 `run_id`로 순서대로 실행된다.

Teacher와 Student의 성능 차이를 키운 `Qwen 7B → 1.5B` 비교는 동일 조건의 별도 설정으로 실행한다.

```bash
scripts/start_azure_h100_job.sh configs/h100_qwen7b_korean_4way.yaml
scripts/monitor_azure_h100.sh --watch
```

H100 VM은 Spot 인스턴스이며 `/mnt`는 eviction 시 초기화된다. 7B 설정은 Teacher
전체 checkpoint 대신 LoRA adapter를 사용하고, adapter·Student checkpoint·로그를
영구 OS 디스크 `/var/lib/kd-results`에 저장한다. 재시작 후 아래 명령을 사용하면
완료된 단계를 건너뛰고 중단 단계부터 다시 실행한다.

```bash
AZURE_H100_RUNNER=scripts/run_azure_h100_resumable.sh \
	scripts/start_azure_h100_job.sh configs/h100_qwen7b_korean_4way.yaml
```

장시간 실험은 Mac에서 watchdog을 실행하는 것을 권장한다. VM이 회수되면 시작을
재시도하고, `/mnt` 환경을 다시 설치하고, 영구 OS 디스크의 checkpoint를 기준으로
미완료 단계부터 재개한다. 완료되면 JSON 로그를 로컬 `results`로 회수하고 차트를
생성한다.

```bash
caffeinate -i .venv/bin/python scripts/watch_azure_h100_job.py \
	configs/h100_qwen7b_korean_4way.yaml
```

7B 첫 결과에서 `batchmean` KL이 CE보다 약 100배 커져 KD가 과도하게 지배했다.
다음 실험은 기존 Teacher adapter와 FT baseline을 재사용하고 Student KD만 다시
학습한다. Temperature를 4로 높이고 KL을 유효 token 평균으로 정규화한다.

```bash
AZURE_H100_RUNNER=scripts/run_azure_h100_kd_retry.sh \
	scripts/start_azure_h100_job.sh configs/h100_qwen7b_korean_t4_tokenmean.yaml
```

Spot 자동 복구와 로컬 결과 회수까지 포함하려면 watchdog에 retry runner를 지정한다.

```bash
caffeinate -i .venv/bin/python scripts/watch_azure_h100_job.py \
	configs/h100_qwen7b_korean_t4_tokenmean.yaml \
	--runner scripts/run_azure_h100_kd_retry.sh
```

Azure Run Command는 동기 실행 중 다른 상태 조회를 막는다. Mac에서 실시간 진행률을 보려면 학습을 백그라운드 작업으로 시작한다.

```bash
scripts/start_azure_h100_job.sh configs/h100_qwen_korean.yaml --step distill
```

출력된 원격 로그 경로를 사용하거나 가장 최근 작업을 자동 선택해 상태를 확인한다.

```bash
scripts/monitor_azure_h100.sh --watch
```

모니터는 30초마다 H100 메모리·사용률과 원격 로그의 최근 tqdm 진행률을 갱신한다. `INTERVAL=10`처럼 조회 간격을 바꿀 수 있다.

## 세 모델 본실험

```bash
bash scripts/run_azure_h100.sh configs/h100_qwen_korean.yaml --step distill
bash scripts/run_azure_h100.sh configs/h100_polyglot_korean.yaml --step distill
bash scripts/run_azure_h100.sh configs/h100_llama_korean.yaml --step distill
```

Llama는 Hugging Face 모델 사용 승인을 받은 뒤 `HF_TOKEN`을 환경 변수로 전달해야 한다.