"""Token Analysis — 각 position 의 top-k 분포와 Teacher vs Student KL divergence 시각화"""

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
from ui.frontend.utils.prompts import PROMPT_TEMPLATES
from ui.frontend.utils.state import MODEL_DISPLAY, active_model_ids, init_session

st.set_page_config(page_title="Token Analysis", page_icon="🔬", layout="wide")
init_session()

st.title("🔬 Token Analysis")
st.caption("prompt 의 각 위치에서 모델별 next-token 분포를 비교합니다. "
             "teacher-forcing 방식으로 정답을 넣고 확률만 뽑습니다.")

run_id = st.session_state.get("selected_run_id")
if not run_id:
    st.warning("🏠 사이드바의 **홈** 에서 Run 을 먼저 선택하고 모델을 로드해 주세요.")
    st.stop()

st.caption(f"Run: `{run_id}`")

# ----- 입력 -----
col_p, col_c = st.columns([2, 1])

with col_p:
    def _on_tok_template():
        tpl = st.session_state["tok_template"]
        txt = PROMPT_TEMPLATES[tpl]
        if txt:
            st.session_state["tok_prompt"] = txt

    st.selectbox(
        "프롬프트 템플릿",
        list(PROMPT_TEMPLATES.keys()),
        index=0,
        key="tok_template",
        on_change=_on_tok_template,
    )
    prompt = st.text_area(
        "분석할 Prompt (길수록 정보 많지만 느려짐)",
        height=120,
        key="tok_prompt",
        placeholder="The capital of France is Paris.",
    )

with col_c:
    st.markdown("**분석 파라미터**")
    model_opts = active_model_ids()
    teacher_id = st.selectbox(
        "Teacher (기준)",
        options=model_opts,
        index=model_opts.index("teacher_ft") if "teacher_ft" in model_opts else 0,
        help="KL divergence 의 P 분포(기준)",
    )
    top_k = st.slider("Top-K", 3, 20, 10)
    temperature = st.slider("분포 온도 T", 0.1, 5.0, 1.0, step=0.1,
                             help="softmax 전 logits 을 T 로 나눔. T=1 이 원 분포.")
    run_btn = st.button("🔎 분석 실행", type="primary", use_container_width=True)

st.divider()

# ----- 실행 -----
if run_btn:
    if not prompt.strip():
        st.error("prompt 가 비어 있습니다.")
        st.stop()
    with st.spinner("분석 중..."):
        try:
            data = api_client.token_analysis(
                run_id=run_id,
                model_ids=active_model_ids(),
                prompt=prompt,
                top_k=top_k,
                temperature=temperature,
                teacher_id=teacher_id,
            )
            st.session_state["tok_result"] = data
        except Exception as e:  # noqa: BLE001
            st.error(f"분석 실패: {e}")
            st.stop()

data = st.session_state.get("tok_result")
if not data:
    st.info("prompt 입력 후 **분석 실행** 버튼을 눌러주세요.")
    st.stop()

if "warning" in data:
    st.warning(data["warning"])

tokens = data["tokens"]
per_model = data["per_model"]
kl_all = data.get("kl_divergence", {})
n_pos = len(tokens) - 1

st.caption(f"토큰 {len(tokens)}개 · 분석 position {n_pos}개 · T={data['temperature']} · top-k={data['top_k']}")

# ----- 요약: KL divergence 전체 -----
st.markdown("### 📏 KL divergence 요약")
if not kl_all:
    st.info("Teacher 모델이 로드되지 않아 KL divergence 를 계산하지 못했습니다.")
else:
    kl_summary_cols = st.columns(len(kl_all))
    for col, (name, values) in zip(kl_summary_cols, kl_all.items()):
        if not values:
            continue
        mean_kl = sum(values) / len(values)
        max_kl = max(values)
        col.metric(
            name.replace(f"{data.get('teacher_id', '')}_vs_", "").replace("_vs_", " vs "),
            f"{mean_kl:.4f}",
            help=f"max {max_kl:.4f}",
        )

    # position 별 KL 라인 차트
    kl_rows = []
    for name, values in kl_all.items():
        short = name.split("_vs_")[-1]
        for pos, v in enumerate(values):
            kl_rows.append({
                "position": pos,
                "token": tokens[pos],
                "pair": short,
                "kl": v,
            })
    if kl_rows:
        df_kl = pd.DataFrame(kl_rows)
        fig = px.line(
            df_kl, x="position", y="kl", color="pair",
            markers=True, hover_data=["token"],
            title=f"position 별 KL(Teacher || Student) · T={data['temperature']}",
            labels={"position": "prompt 내 position", "kl": "KL divergence (nats)"},
        )
        fig.update_layout(height=380, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ----- Position 별 Top-K 분포 -----
st.markdown("### 🎯 Position 별 Top-K 분포")

if n_pos <= 0:
    st.stop()

# 토큰 선택 (버튼 스트립 대신 slider + selectbox)
sel_col1, sel_col2 = st.columns([3, 2])
with sel_col1:
    pos = st.slider("분석할 position", 0, n_pos - 1, 0,
                     help=f"0 = 첫 토큰 다음 예측, {n_pos - 1} = 마지막 정답 예측")
with sel_col2:
    st.caption("prompt 토큰들 (선택된 위치 강조)")
    # 토큰 시각화
    parts = []
    for i, t in enumerate(tokens):
        display = t.replace(" ", "·").replace("\n", "↵")
        if i == pos:
            parts.append(f"**`[{display}]`**")
        elif i == pos + 1:
            parts.append(f"🎯`{display}`")
        else:
            parts.append(f"`{display}`")
    st.markdown(" ".join(parts))

# 선택된 position 의 top-k 분포
st.markdown(f"#### Position {pos}: `{tokens[pos]}` → 다음 토큰 예측")
actual_next = tokens[pos + 1]
st.caption(f"🎯 실제 다음 토큰: `{actual_next}`")

cols = st.columns(max(sum(1 for m in per_model if "error" not in per_model[m]), 1))

col_iter = iter(cols)
for mid, m_data in per_model.items():
    if "error" in m_data:
        st.error(f"{mid}: {m_data['error']}")
        continue
    positions = m_data.get("positions", [])
    if pos >= len(positions):
        continue
    p = positions[pos]

    try:
        col = next(col_iter)
    except StopIteration:
        break

    with col:
        st.markdown(f"**{MODEL_DISPLAY.get(mid, mid)}**")
        top_tokens = p["top_tokens"]
        top_probs = p["top_probs"]
        actual_id = p["actual_next_id"]

        # 바 차트 (내림차순)
        display_tokens = [
            t.replace(" ", "·").replace("\n", "↵") or "␣"
            for t in top_tokens
        ]
        colors = [
            "rgb(239,68,68)" if tid == actual_id else "rgb(99,102,241)"
            for tid in p["top_ids"]
        ]
        fig = go.Figure(go.Bar(
            x=top_probs[::-1],
            y=display_tokens[::-1],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:.3f}" for v in top_probs[::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis=dict(range=[0, max(top_probs) * 1.25], title="prob"),
            yaxis=dict(title=""),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 정답이 top-k 안에 있는지
        if actual_id in p["top_ids"]:
            rank = p["top_ids"].index(actual_id) + 1
            hit_prob = top_probs[p["top_ids"].index(actual_id)]
            st.success(f"✅ 정답 rank **{rank}** (p={hit_prob:.3f})")
        else:
            st.warning(f"⚠️ 정답 `{actual_next}` 은 top-{top_k} 밖")

# ----- 원본 데이터 -----
with st.expander("원본 분석 결과 (JSON)"):
    st.json(data)
