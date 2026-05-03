"""Session state \ud5ec\ud37c"""

from __future__ import annotations

import streamlit as st

DEFAULT_MODEL_IDS = ["teacher_ft", "student_kd", "student_ft"]
MODEL_DISPLAY = {
    "teacher_ft": "Teacher (FT)",
    "student_kd": "Student (KD)",
    "student_ft": "Student (FT)",
    "student_base": "Student (Base)",
}


def init_session():
    st.session_state.setdefault("selected_run_id", None)
    st.session_state.setdefault("include_base", False)
    st.session_state.setdefault("last_generation", None)
    st.session_state.setdefault("history", [])


def active_model_ids() -> list[str]:
    ids = list(DEFAULT_MODEL_IDS)
    if st.session_state.get("include_base"):
        ids.append("student_base")
    return ids
