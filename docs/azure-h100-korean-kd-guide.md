# Azure H100 한국어 지식 증류 가이드

대상 VM은 단일 `NVIDIA H100 NVL 94GB`이므로 DDP를 사용하지 않는다. `python main.py ...`로 실행하면 기존 코드는 자동으로 단일 프로세스 경로를 사용한다.

## 저장 경로

- 소스: `/mnt/knowledge-distillation`
- Python 환경: `/mnt/kd-venv`
- Hugging Face 모델/데이터 캐시: `/mnt/huggingface`
- 결과: `/mnt/kd-results/<model>`

OS 디스크가 아닌 `/mnt`의 120GB 임시 디스크를 사용한다. Azure 임시 디스크는 VM 재배포 또는 호스트 이동 시 유실될 수 있으므로 완료된 결과는 즉시 Blob Storage나 로컬로 회수해야 한다.

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