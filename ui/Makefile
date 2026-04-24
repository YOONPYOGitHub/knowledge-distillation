.PHONY: backend frontend up down install

install:
	uv sync --extra ui

backend:
	cd .. && uv run uvicorn ui.backend.app:app --reload --host 127.0.0.1 --port 8000

frontend:
	cd .. && uv run streamlit run ui/frontend/app.py --server.port 8501

up:
	@echo "백엔드/프론트엔드를 분리된 터미널에서 실행하세요:"
	@echo "  터미널 1: cd ui && make backend"
	@echo "  터미널 2: cd ui && make frontend"
	@echo ""
	@echo "또는 다음 명령을 직접 사용하세요:"
	@echo "  (uv run uvicorn ui.backend.app:app --reload --port 8000 &) && uv run streamlit run ui/frontend/app.py --server.port 8501"
