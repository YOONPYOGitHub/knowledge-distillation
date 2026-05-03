"""Streamlit 엔트리 — st.navigation 기반 멀티페이지 라우팅.

사이드바는 **네비게이션 전용**. 설정(Run 선택 / 모델 로드)은 홈 본문에 배치.

실행:
  uv run streamlit run ui/frontend/app.py --server.port 8501
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="KD Comparison",
    page_icon="🧪",
    layout="wide",
)

_PAGES_DIR = Path(__file__).parent / "pages"

# dict 의 "" 키는 그룹 헤더 없이 표시됨 → "홈 > 홈" 중복 방지.
_nav = st.navigation(
    {
        "": [
            st.Page(
                Path(__file__).parent / "home.py",
                title="홈",
                icon="🏠",
                default=True,
                url_path="home",
            ),
        ],
        "🧪 생성": [
            st.Page(
                _PAGES_DIR / "1_Compare_Generate.py",
                title="Compare Generate",
                icon="🆚",
                url_path="compare",
            ),
            st.Page(
                _PAGES_DIR / "7_Batch_Prompts.py",
                title="Batch Prompts",
                icon="📦",
                url_path="batch",
            ),
        ],
        "📝 평가": [
            st.Page(
                _PAGES_DIR / "4_Blind_Evaluation.py",
                title="Blind Evaluation",
                icon="🕶️",
                url_path="blind",
            ),
            st.Page(
                _PAGES_DIR / "5_Aggregation.py",
                title="Aggregation",
                icon="📈",
                url_path="aggregate",
            ),
        ],
        "🔬 분석": [
            st.Page(
                _PAGES_DIR / "2_Experiment_Report.py",
                title="Experiment Report",
                icon="📊",
                url_path="report",
            ),
            st.Page(
                _PAGES_DIR / "6_Token_Analysis.py",
                title="Token Analysis",
                icon="🔬",
                url_path="tokens",
            ),
            st.Page(
                _PAGES_DIR / "3_Checkpoint_Browser.py",
                title="Checkpoint Browser",
                icon="💾",
                url_path="checkpoints",
            ),
        ],
    }
)

_nav.run()
