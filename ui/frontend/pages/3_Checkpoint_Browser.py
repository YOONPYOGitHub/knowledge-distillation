"""Checkpoint Browser — run 별 체크포인트 메타데이터 탐색 & 로드"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datetime import datetime

import streamlit as st

from ui.frontend.utils import api_client
from ui.frontend.utils.state import MODEL_DISPLAY, active_model_ids, init_session

st.set_page_config(page_title="Checkpoint Browser", page_icon="💾", layout="wide")
init_session()

st.title("💾 Checkpoint Browser")
st.caption("저장된 모든 run 의 체크포인트를 탐색하고, 선택한 run 으로 모델을 로드할 수 있습니다.")

try:
    runs = api_client.list_runs()
except Exception as e:  # noqa: BLE001
    st.error(f"run 목록 로드 실패: {e}")
    st.stop()

if not runs:
    st.warning("저장된 run 이 없습니다.")
    st.stop()

# --- 필터 ---
c1, c2 = st.columns([1, 1])
with c1:
    search = st.text_input("검색 (run_id 일부)", "")
with c2:
    only_full = st.toggle(
        "3종 체크포인트 모두 있는 run 만",
        value=False,
        help="teacher_ft / student_kd / student_ft 모두 있어야 증류 비교 가능",
    )

selected_run_id = st.session_state.get("selected_run_id")

# --- run 카드 ---
filtered = []
for r in runs:
    if search and search not in r["run_id"]:
        continue
    ckpts = r.get("checkpoints", {})
    if only_full and not all(ckpts.get(k) for k in ("teacher_ft", "student_kd", "student_ft")):
        continue
    filtered.append(r)

st.caption(f"총 **{len(filtered)}** 개 run 표시 중")
st.divider()

# 2열 그리드
cols_per_row = 2
for row_start in range(0, len(filtered), cols_per_row):
    row = filtered[row_start : row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for col, r in zip(cols, row):
        with col:
            rid = r["run_id"]
            is_selected = rid == selected_run_id
            marker = "🟢 **현재 선택됨**" if is_selected else ""
            with st.container(border=True):
                st.markdown(f"### `{rid}` {marker}")
                st.caption(
                    f"생성: {r.get('created_at', '-')} · "
                    f"Teacher: `{r['teacher_model']}` · Student: `{r['student_model']}`"
                )

                # 체크포인트 존재 뱃지
                ckpts = r.get("checkpoints", {})
                badges = []
                for k in ("teacher_ft", "student_kd", "student_ft", "student_base"):
                    mark = "✅" if ckpts.get(k) else "❌"
                    badges.append(f"{mark} {k}")
                st.markdown(" · ".join(badges))

                # 하이퍼파라미터
                hp = r.get("hyperparams") or {}
                if hp:
                    with st.expander("Hyperparameters"):
                        st.json(hp)

                # PPL 요약
                results = r.get("results") or {}
                if results:
                    rows = []
                    for mid, v in results.items():
                        rows.append({
                            "model": MODEL_DISPLAY.get(mid, mid),
                            "PPL": (f"{v.get('ppl'):.2f}"
                                    if v.get("ppl") is not None else "-"),
                            "tok/s": (f"{v.get('tokens_per_sec'):.0f}"
                                      if v.get("tokens_per_sec") is not None else "-"),
                        })
                    st.dataframe(rows, hide_index=True, use_container_width=True)

                # 버튼
                action_cols = st.columns(2)
                with action_cols[0]:
                    if st.button("현재 Run 으로 설정", key=f"select_{rid}",
                                  use_container_width=True,
                                  disabled=is_selected):
                        st.session_state["selected_run_id"] = rid
                        st.rerun()
                with action_cols[1]:
                    if st.button("모델 로드", key=f"load_{rid}",
                                  use_container_width=True, type="primary"):
                        st.session_state["selected_run_id"] = rid
                        with st.spinner(f"{rid} 모델 로드 중..."):
                            try:
                                resp = api_client.load_models(rid, active_model_ids())
                                st.success(
                                    f"로드 완료: {resp['loaded']} · "
                                    f"{resp['memory_mb']:.0f} MB"
                                )
                            except Exception as e:  # noqa: BLE001
                                st.error(f"실패: {e}")

if not filtered:
    st.info("조건에 맞는 run 이 없습니다.")
