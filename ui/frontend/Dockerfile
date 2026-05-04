# Streamlit frontend for Azure Container Apps.
# 빌드 컨텍스트는 프로젝트 루트 기준.
#   docker build -f ui/frontend/Dockerfile -t kd-frontend .

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN pip install \
    "streamlit>=1.32.0" \
    "plotly>=5.18.0" \
    "httpx>=0.27.0" \
    "python-dotenv>=1.0.0" \
    "pandas>=2.0.0"

COPY ui/__init__.py /app/ui/__init__.py
COPY ui/frontend /app/ui/frontend

# 기본값. 실제 백엔드 URL 은 Container Apps 환경변수로 주입.
ENV API_BASE_URL=http://127.0.0.1:8000 \
    PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "ui/frontend/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
