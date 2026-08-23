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

## 세 모델 본실험

```bash
bash scripts/run_azure_h100.sh configs/h100_qwen_korean.yaml --step distill
bash scripts/run_azure_h100.sh configs/h100_polyglot_korean.yaml --step distill
bash scripts/run_azure_h100.sh configs/h100_llama_korean.yaml --step distill
```

Llama는 Hugging Face 모델 사용 승인을 받은 뒤 `HF_TOKEN`을 환경 변수로 전달해야 한다.