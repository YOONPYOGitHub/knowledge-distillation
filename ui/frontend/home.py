"""홈 — 설정 컨트롤 센터 (Run 선택 / 모델 로드) + 상태 요약."""

from __future__ import annotations

import streamlit as st

from ui.frontend.utils import api_client
from ui.frontend.utils.state import (
    MODEL_DISPLAY,
    active_model_ids,
    init_session,
)

init_session()

# ── 헤더 ────────────────────────────────────────────────────────
hdr_l, hdr_r = st.columns([6, 4])
with hdr_l:
    st.title("🧪 KD Comparison")
    st.caption(
        "Teacher(FT) / Student(KD) / Student(FT) / Student(Base) 모델을 "
        "동일 prompt 로 비교 생성하고, 정성·정량 평가를 수집합니다."
    )

health = api_client.health()
with hdr_r:
    st.markdown(" ")
    if health is not None:
        st.success(f"✅ Backend OK  ·  device `{health['device']}`", icon="🟢")
    else:
        st.error("❌ Backend 연결 실패 — `make backend`", icon="🔴")
        st.stop()

# ── 설정: Run 선택 ──────────────────────────────────────────────
try:
    runs = api_client.list_runs()
except Exception as e:  # noqa: BLE001
    st.error(f"Run 목록 로드 실패: {e}")
    st.stop()

if not runs:
    st.warning("학습된 run 이 없습니다.")
    st.caption(
        "먼저 파이프라인 실행:\n```\nuv run python main.py "
        "configs/exp08_alpha02_epoch8.yaml\n```"
    )
    st.stop()

st.subheader("⚙️ 설정")

cfg_l, cfg_r = st.columns([3, 2])

with cfg_l:
    run_ids = [r["run_id"] for r in runs]
    current = st.session_state.get("selected_run_id")
    default_idx = run_ids.index(current) if current in run_ids else 0

    selected = st.selectbox(
        "Run ID",
        run_ids,
        index=default_idx,
        help="results/logs/ 에서 자동 스캔",
        key="home_run_select",
    )
    st.session_state["selected_run_id"] = selected

    include_base = st.toggle(
        "Student (Base) 포함 (4열)",
        value=st.session_state.get("include_base", False),
        help="끄면 3열(Teacher FT / KD / FT). 켜면 Base 추가.",
        key="home_include_base",
    )
    st.session_state["include_base"] = include_base

    if st.button(
        "모델 로드",
        use_container_width=True,
        type="primary",
    ):
        with st.spinner("모델 로드 중..."):
            try:
                resp = api_client.load_models(selected, active_model_ids())
                st.session_state["last_loaded_models"] = resp["loaded"]
                st.session_state["last_loaded_run"] = selected
                st.success(
                    f"로드 완료: {resp['loaded']} · "
                    f"memory={resp['memory_mb']:.1f} MB · "
                    f"device={resp['device']}"
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"로드 실패: {e}")

with cfg_r:
    info = next((r for r in runs if r["run_id"] == selected), None)
    if info:
        with st.container(border=True):
            st.caption("**선택된 Run 정보**")
            st.markdown(
                f"- Teacher: `{info['teacher_model']}`  \n"
                f"- Student: `{info['student_model']}`  \n"
                f"- 생성: {info.get('created_at', '-')}"
            )
            missing = [k for k, v in info["checkpoints"].items() if not v]
            if missing:
                st.warning(f"누락된 체크포인트: {missing}")
            hp = info.get("hyperparams") or {}
            if hp:
                with st.expander("하이퍼파라미터"):
                    st.json(hp, expanded=True)

st.divider()

# ── 빠른 시작 상태 ──────────────────────────────────────────────
loaded_models = st.session_state.get("last_loaded_models") or []
loaded_run = st.session_state.get("last_loaded_run")
is_loaded = bool(loaded_models) and loaded_run == selected

st.subheader("🚀 빠른 시작")

qs_cols = st.columns([1, 1, 1, 2])
with qs_cols[0]:
    st.metric(
        "Run",
        selected,
        delta="✅ 선택됨",
    )
with qs_cols[1]:
    st.metric(
        "모델",
        f"{len(loaded_models)}개 로드" if is_loaded else "—",
        delta="✅" if is_loaded else "⬜ 위에서 로드",
        delta_color="normal" if is_loaded else "off",
    )
with qs_cols[2]:
    st.metric("Device", health["device"])
with qs_cols[3]:
    st.markdown(" ")
    st.page_link(
        "pages/1_Compare_Generate.py",
        label="➡️  Compare Generate 로 이동",
        use_container_width=True,
    )

st.divider()

# ── 페이지 안내 (간단 리스트) ───────────────────────────────────
st.subheader("📑 페이지 안내")

g1, g2, g3 = st.columns(3)
with g1:
    st.markdown("**🧪 생성**")
    st.caption("모델이 어떻게 뱉는지 직접 확인")
    st.page_link("pages/1_Compare_Generate.py", label="🆚 Compare Generate")
    st.page_link("pages/7_Batch_Prompts.py",    label="📦 Batch Prompts")
with g2:
    st.markdown("**📝 평가**")
    st.caption("사람의 판단을 수집·집계")
    st.page_link("pages/4_Blind_Evaluation.py", label="🕶️ Blind Evaluation")
    st.page_link("pages/5_Aggregation.py",      label="📈 Aggregation")
with g3:
    st.markdown("**🔬 분석**")
    st.caption("정량 지표·실험 메타로 결론")
    st.page_link("pages/2_Experiment_Report.py",  label="📊 Experiment Report")
    st.page_link("pages/6_Token_Analysis.py",     label="🔬 Token Analysis")
    st.page_link("pages/3_Checkpoint_Browser.py", label="💾 Checkpoint Browser")

# ── 부가 정보 ────────────────────────────────────────────────────
st.divider()
with st.expander("📘 모델 네이밍 규칙"):
    for mid, name in MODEL_DISPLAY.items():
        desc = {
            "teacher_ft": "도메인 적응된 Teacher (Upper bound)",
            "student_kd": "지식 증류로 학습된 Student ★ 핵심",
            "student_ft": "CE loss 만으로 학습된 Student (대조군)",
            "student_base": "Pretrained distilgpt2 원본 (Lower bound)",
        }.get(mid, "")
        st.markdown(f"- `{mid}` — **{name}** · {desc}")
