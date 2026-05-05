"""Experiment Report — 여러 run 의 loss 곡선 / PPL / 하이퍼파라미터 스캔 대시보드"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui.frontend.utils import api_client
from ui.frontend.utils.state import MODEL_DISPLAY, init_session

st.set_page_config(page_title="Experiment Report", page_icon="📊", layout="wide")
init_session()

st.title("📊 Experiment Report")
st.caption("여러 실험(run)의 학습 곡선과 평가 결과를 한 곳에서 비교합니다.")

# --- Run 목록 ---
try:
    runs = api_client.list_runs()
except Exception as e:  # noqa: BLE001
    st.error(f"run 목록 로드 실패: {e}")
    st.stop()

if not runs:
    st.warning("학습된 run 이 없습니다. `main.py` 로 파이프라인을 먼저 실행하세요.")
    st.stop()


# --- Run 멀티셀렉트 ---
run_ids = [r["run_id"] for r in runs]
run_by_id = {r["run_id"]: r for r in runs}

default_ids = run_ids[:3]  # 최신 3개
selected_ids = st.multiselect(
    "비교할 Run 선택",
    options=run_ids,
    default=default_ids,
    help="여러 run 을 겹쳐서 비교할 수 있습니다.",
)

if not selected_ids:
    st.info("비교할 run 을 하나 이상 선택해 주세요.")
    st.stop()

# --- 선택된 run 요약 테이블 ---
summary_rows = []
for rid in selected_ids:
    info = run_by_id[rid]
    row = {
        "run_id": rid,
        "teacher": info["teacher_model"],
        "student": info["student_model"],
        **info.get("hyperparams", {}),
    }
    for mid, r in info.get("results", {}).items():
        row[f"PPL_{mid}"] = r.get("ppl")
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
st.dataframe(df_summary, use_container_width=True, hide_index=True)
st.divider()


# --- 데이터 캐싱 헬퍼 ---
@st.cache_data(show_spinner=False)
def _fetch_history(run_id: str) -> dict:
    return api_client.get_history(run_id)


@st.cache_data(show_spinner=False)
def _fetch_evaluation(run_id: str) -> list[dict]:
    try:
        return api_client.get_evaluation(run_id)
    except Exception as e:  # noqa: BLE001
        st.warning(f"{run_id}: evaluation 조회 실패 ({e})")
        return []


@st.cache_data(show_spinner=False)
def _fetch_figures(run_id: str) -> list[dict]:
    return api_client.list_figures(run_id)


# --- 탭 구조 ---
tab_loss, tab_ppl, tab_speed, tab_scan, tab_figures, tab_diff = st.tabs(
    ["📉 Loss 곡선", "🎯 PPL 비교", "⚡ 속도 & 메모리", "🔍 하이퍼파라미터 스캔",
     "🖼️ 기존 차트 (PNG)", "🆚 Run Diff"]
)


# ============ Loss 곡선 ============
with tab_loss:
    loss_type = st.radio(
        "Loss 종류",
        ["distill", "baseline", "teacher"],
        horizontal=True,
    )

    loss_key_options = {
        "distill": ["total_loss", "ce_loss", "kd_loss", "val_total_loss", "val_ce_loss"],
        "baseline": ["ce_loss", "val_ce_loss"],
        "teacher": ["ce_loss", "val_ce_loss"],
    }
    metric = st.selectbox("Metric", loss_key_options[loss_type])

    fig = go.Figure()
    any_data = False
    for rid in selected_ids:
        try:
            hist = _fetch_history(rid)
        except Exception as e:  # noqa: BLE001
            st.warning(f"{rid}: history 로드 실패 ({e})")
            continue
        records = hist.get(loss_type)
        if not records:
            continue
        any_data = True
        epochs = [r.get("epoch", i + 1) for i, r in enumerate(records)]
        values = [r.get(metric) for r in records]
        fig.add_trace(go.Scatter(
            x=epochs, y=values, mode="lines+markers", name=rid,
        ))

    if any_data:
        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title=metric,
            height=450,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"선택된 run 들에 `{loss_type}` history 가 없습니다.")


# ============ PPL 비교 ============
with tab_ppl:
    st.caption("각 run 의 4-Way 평가 결과를 막대그래프로 비교")

    rows = []
    for rid in selected_ids:
        evals = _fetch_evaluation(rid)
        for e in evals:
            rows.append({
                "run_id": rid,
                "model": e.get("name"),
                "ppl": e.get("perplexity"),
                "tokens_per_sec": e.get("tokens_per_sec"),
                "params_M": (e.get("total_params") or 0) / 1e6,
            })

    if not rows:
        st.info("evaluation_results.json 이 없는 run 만 선택되었습니다.")
    else:
        df_eval = pd.DataFrame(rows)

        fig_ppl = px.bar(
            df_eval, x="model", y="ppl", color="run_id", barmode="group",
            title="Perplexity (↓ 낮을수록 좋음)",
            labels={"ppl": "PPL"},
        )
        st.plotly_chart(fig_ppl, use_container_width=True)

        # Q1/Q2/Q3 카드 (첫 번째 run 기준)
        first_run = selected_ids[0]
        first_eval = {e["name"]: e for e in _fetch_evaluation(first_run)}

        def _ppl(name):
            e = first_eval.get(name)
            return e.get("perplexity") if e else None

        teacher_ppl = _ppl("Teacher (FT)") or _ppl("Teacher")
        kd_ppl = _ppl("Student (KD)")
        ft_ppl = _ppl("Student (FT)")
        base_ppl = _ppl("Student (Base)")

        st.markdown(f"**Q1/Q2/Q3 — run: `{first_run}` 기준**")
        c1, c2, c3 = st.columns(3)
        if teacher_ppl and kd_ppl:
            diff = kd_ppl - teacher_ppl
            c1.metric("Q1. 압축 손실 (KD − Teacher)", f"{diff:+.2f}",
                      help="Student 가 Teacher 대비 얼마나 손실")
        if kd_ppl and ft_ppl:
            delta = ft_ppl - kd_ppl
            pct = delta / ft_ppl * 100 if ft_ppl else 0
            c2.metric("Q2. KD 효과 (FT − KD)", f"{delta:+.2f}",
                      delta=f"{pct:+.1f}%", help="양수면 KD 가 더 우수")
        if ft_ppl and base_ppl:
            delta = base_ppl - ft_ppl
            c3.metric("Q3. FT 효과 (Base − FT)", f"{delta:+.2f}",
                      help="양수면 FT 가 Base 대비 우수")

        with st.expander("원본 데이터"):
            st.dataframe(df_eval, use_container_width=True, hide_index=True)


# ============ 속도 & 메모리 ============
with tab_speed:
    rows = []
    for rid in selected_ids:
        for e in _fetch_evaluation(rid):
            rows.append({
                "run_id": rid,
                "model": e.get("name"),
                "tokens_per_sec": e.get("tokens_per_sec"),
                "ms_per_token": e.get("ms_per_token"),
                "size_mb": e.get("size_mb"),
                "params_M": (e.get("total_params") or 0) / 1e6,
            })

    if not rows:
        st.info("평가 결과가 없습니다.")
    else:
        df = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(df, x="model", y="tokens_per_sec", color="run_id",
                         barmode="group", title="tokens/sec (↑ 높을수록 좋음)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(df, x="model", y="size_mb", color="run_id",
                         barmode="group", title="모델 크기 (MB)")
            st.plotly_chart(fig, use_container_width=True)


# ============ 하이퍼파라미터 스캔 ============
with tab_scan:
    st.caption("전체 run 중 KD 체크포인트가 있는 것들만 모아 (α, T) vs PPL 을 분석")

    points = []
    for r in runs:
        hp = r.get("hyperparams") or {}
        res = r.get("results") or {}
        kd = res.get("student_kd")
        if not kd or "alpha" not in hp or "temperature" not in hp:
            continue
        points.append({
            "run_id": r["run_id"],
            "alpha": hp["alpha"],
            "temperature": hp["temperature"],
            "epochs": hp.get("epochs"),
            "kd_ppl": kd.get("ppl"),
        })

    if not points:
        st.info("α, T, KD PPL 이 모두 있는 run 이 없습니다.")
    else:
        df_scan = pd.DataFrame(points)
        fig = px.scatter(
            df_scan, x="alpha", y="temperature", size="epochs", color="kd_ppl",
            hover_data=["run_id", "kd_ppl"],
            color_continuous_scale="Viridis_r",
            title="α vs T (색=KD PPL, 점 크기=epochs)",
        )
        # 최솟값 하이라이트
        best = df_scan.loc[df_scan["kd_ppl"].idxmin()]
        fig.add_trace(go.Scatter(
            x=[best["alpha"]], y=[best["temperature"]],
            mode="markers",
            marker=dict(size=20, symbol="star", color="red"),
            name=f"best: {best['run_id']} (PPL {best['kd_ppl']:.2f})",
        ))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df_scan.sort_values("kd_ppl").reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )


# ============ 기존 PNG 차트 ============
with tab_figures:
    st.caption("`compare.py` 가 생성한 기존 PNG 차트를 그대로 임베드")
    for rid in selected_ids:
        figs = _fetch_figures(rid)
        if not figs:
            continue
        st.markdown(f"#### `{rid}`")
        cols = st.columns(min(len(figs), 2))
        for i, f in enumerate(figs):
            with cols[i % len(cols)]:
                try:
                    img = api_client.figure_bytes(rid, f["filename"])
                    st.image(img, caption=f["filename"])
                except Exception as e:
                    st.error(f"로드 실패: {f['filename']} ({e})")


# ============ Run Diff ============
with tab_diff:
    if len(selected_ids) < 2:
        st.info("최소 2개 run 을 선택해야 비교할 수 있습니다.")
    else:
        c1, c2 = st.columns(2)
        rid_a = c1.selectbox("Run A", selected_ids, index=0, key="diff_a")
        rid_b = c2.selectbox("Run B", selected_ids,
                              index=min(1, len(selected_ids) - 1), key="diff_b")

        a = run_by_id[rid_a]
        b = run_by_id[rid_b]

        hp_a = a.get("hyperparams", {})
        hp_b = b.get("hyperparams", {})
        all_keys = sorted(set(hp_a) | set(hp_b))
        hp_rows = [
            {"key": k, "Run A": hp_a.get(k, "-"), "Run B": hp_b.get(k, "-"),
             "diff": "✅" if hp_a.get(k) == hp_b.get(k) else "⚠️"}
            for k in all_keys
        ]
        st.markdown("**하이퍼파라미터**")
        st.dataframe(pd.DataFrame(hp_rows), hide_index=True, use_container_width=True)

        res_a = a.get("results", {})
        res_b = b.get("results", {})
        all_mids = sorted(set(res_a) | set(res_b))
        res_rows = []
        for m in all_mids:
            ppl_a = (res_a.get(m) or {}).get("ppl")
            ppl_b = (res_b.get(m) or {}).get("ppl")
            delta = None
            if ppl_a is not None and ppl_b is not None:
                delta = ppl_b - ppl_a
            res_rows.append({
                "model": MODEL_DISPLAY.get(m, m),
                f"PPL ({rid_a})": ppl_a,
                f"PPL ({rid_b})": ppl_b,
                "Δ (B − A)": delta,
            })
        st.markdown("**PPL 비교**")
        st.dataframe(pd.DataFrame(res_rows), hide_index=True, use_container_width=True)
