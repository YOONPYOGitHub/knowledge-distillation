"""Aggregation — 저장된 feedback JSONL 들을 모아 집계 대시보드

- compare 모드: ratings/preference 에서 model_id 그대로 사용
- blind 모드: blind_mapping 으로 A/B/C → 실제 model_id 로 역매핑 후 집계
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from ui.frontend.utils import api_client
from ui.frontend.utils.state import MODEL_DISPLAY, init_session

st.set_page_config(page_title="Aggregation", page_icon="📈", layout="wide")
init_session()

st.title("📈 Feedback Aggregation")
st.caption("서버에 저장된 JSONL feedback 파일들을 모아 정성 평가 결과를 집계합니다.")

try:
    runs = api_client.list_runs()
except Exception as e:  # noqa: BLE001
    st.error(f"run 목록 로드 실패: {e}")
    st.stop()

if not runs:
    st.warning("run 이 없습니다.")
    st.stop()

run_ids = [r["run_id"] for r in runs]
c1, c2 = st.columns([2, 1])
with c1:
    selected_runs = st.multiselect("집계할 Run", run_ids, default=run_ids[:1])
with c2:
    mode_filter = st.selectbox("모드", ["전체", "compare", "blind"], index=0)

if not selected_runs:
    st.info("run 을 하나 이상 선택하세요.")
    st.stop()

mode_arg = None if mode_filter == "전체" else mode_filter

# ----- 데이터 수집 -----
all_entries: list[dict] = []
per_run_count: dict[str, int] = {}
for rid in selected_runs:
    try:
        entries = api_client.read_feedback_entries(rid, mode=mode_arg)
    except Exception as e:  # noqa: BLE001
        st.warning(f"{rid}: 조회 실패 ({e})")
        continue
    # 손상 레코드 필터
    clean = [e for e in entries if "_error" not in e]
    per_run_count[rid] = len(clean)
    all_entries.extend(clean)

if not all_entries:
    st.info("집계할 entry 가 없습니다. 먼저 Compare / Blind 페이지에서 세션을 저장하세요.")
    st.stop()

# ----- 요약 -----
cnt_cols = st.columns(max(len(per_run_count), 1))
for col, (rid, n) in zip(cnt_cols, per_run_count.items()):
    col.metric(rid, f"{n}건")

st.divider()


def _resolve_model(entry: dict, key: str) -> str | None:
    """entry 내 A/B/C 키 또는 model_id 키를 실제 model_id 로 변환"""
    if entry.get("mode") == "blind" and entry.get("blind_mapping"):
        return entry["blind_mapping"].get(key, key)
    return key


# ----- 선호도 집계 -----
pref_counter: Counter = Counter()
for e in all_entries:
    pref = e.get("preference")
    if not pref:
        continue
    if pref == "tie":
        pref_counter["tie"] += 1
    else:
        pref_counter[_resolve_model(e, pref) or pref] += 1

# ----- 리커트 평균 -----
likert_accum: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
for e in all_entries:
    ratings = e.get("ratings") or {}
    for key, vals in ratings.items():
        real = _resolve_model(e, key) or key
        if not isinstance(vals, dict):
            continue
        for metric, v in vals.items():
            if isinstance(v, (int, float)):
                likert_accum[real][metric].append(float(v))

likert_rows = []
for mid, metrics in likert_accum.items():
    row = {"model": MODEL_DISPLAY.get(mid, mid), "model_id": mid,
            "n": max((len(v) for v in metrics.values()), default=0)}
    for metric, lst in metrics.items():
        if lst:
            row[metric] = sum(lst) / len(lst)
    likert_rows.append(row)

df_likert = pd.DataFrame(likert_rows) if likert_rows else pd.DataFrame()

# ----- 탭 -----
tab_pref, tab_likert, tab_raw = st.tabs(["🏆 선호도", "⭐ 리커트 평균", "📋 원본 entries"])

with tab_pref:
    if not pref_counter:
        st.info("선호도 데이터가 없습니다.")
    else:
        rows = [
            {"선택": MODEL_DISPLAY.get(k, k) if k != "tie" else "차이 없음",
             "key": k, "count": v}
            for k, v in pref_counter.most_common()
        ]
        df_pref = pd.DataFrame(rows)
        total = df_pref["count"].sum()
        df_pref["pct"] = (df_pref["count"] / total * 100).round(1)

        c1, c2 = st.columns([2, 1])
        with c1:
            fig = px.bar(
                df_pref, x="선택", y="count", color="선택", text="count",
                title=f"선호도 집계 (전체 {total}건)",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.dataframe(
                df_pref[["선택", "count", "pct"]],
                hide_index=True, use_container_width=True,
            )

with tab_likert:
    if df_likert.empty:
        st.info("리커트 평가 데이터가 없습니다.")
    else:
        metric_cols = [c for c in df_likert.columns
                        if c not in ("model", "model_id", "n")]
        df_display = df_likert[["model", "n", *metric_cols]].copy()
        for m in metric_cols:
            df_display[m] = df_display[m].round(2)
        st.dataframe(df_display, hide_index=True, use_container_width=True)

        # 항목별 막대
        melted = df_likert.melt(
            id_vars=["model"], value_vars=metric_cols,
            var_name="metric", value_name="score",
        ).dropna()
        fig = px.bar(
            melted, x="metric", y="score", color="model", barmode="group",
            title="항목별 평균 점수 (1~5)",
            range_y=[1, 5],
        )
        st.plotly_chart(fig, use_container_width=True)

        # 레이더
        if len(metric_cols) >= 3:
            import plotly.graph_objects as go

            fig_radar = go.Figure()
            for _, row in df_likert.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row.get(m, 0) for m in metric_cols],
                    theta=metric_cols,
                    fill="toself",
                    name=row["model"],
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
                title="모델별 평균 점수 레이더",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

with tab_raw:
    compact = [
        {
            "run_id": e.get("run_id"),
            "mode": e.get("mode"),
            "prompt": (e.get("prompt") or "")[:60],
            "preference": e.get("preference"),
            "source": e.get("_source_file"),
        }
        for e in all_entries
    ]
    st.dataframe(pd.DataFrame(compact), hide_index=True, use_container_width=True)
    with st.expander("전체 JSON 보기 (상위 10건)"):
        st.json(all_entries[:10])
