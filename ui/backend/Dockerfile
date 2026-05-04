# FastAPI backend (CPU-only) for Azure Container Apps.
# 빌드 컨텍스트는 프로젝트 루트(knowledge-distillation/) 기준.
#   docker build -f ui/backend/Dockerfile -t kd-backend .

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# 시스템 의존성 (torch CPU 휠 다운로드 대비)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# torch CPU 휠 (PyTorch CPU index 강제) + 런타임 의존성
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.3.1" \
    && pip install \
        "transformers>=4.35.0,<5.0" \
        "accelerate>=0.24.0,<1.0" \
        "tokenizers>=0.15.0,<0.20" \
        "fastapi>=0.110.0" \
        "uvicorn[standard]>=0.27.0" \
        "pydantic>=2.0" \
        "python-dotenv>=1.0.0" \
        "pyyaml>=6.0" \
        "numpy>=1.24.0,<2.0" \
        "azure-storage-blob>=12.19.0" \
        "azure-identity>=1.15.0"

# 백엔드가 필요로 하는 패키지만 복사 (results/ 는 Blob에서 읽으므로 제외)
COPY ui/__init__.py /app/ui/__init__.py
COPY ui/backend /app/ui/backend
COPY src /app/src

ENV STORAGE_BACKEND=blob \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    AZURE_BLOB_CACHE_DIR=/tmp/kd-cache \
    PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "ui.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
