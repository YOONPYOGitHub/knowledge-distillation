"""Batch Prompts — CSV/텍스트로 여러 prompt 를 한 번에 생성"""

from __future__ import annotations

import io
import json
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from ui.frontend.utils import api_client
from ui.frontend.utils.prompts import PARAM_PRESETS
from ui.frontend.utils.state import MODEL_DISPLAY, active_model_ids, init_session

st.set_page_config(page_title="Batch Prompts", page_icon="📦", layout="wide")
init_session()

st.title("📦 Batch Prompts")
st.caption(
    "여러 prompt 를 한 번에 모든 모델로 생성합니다. 연구용 대량 샘플 수집에 사용하세요."
)

run_id = st.session_state.get("selected_run_id")
if not run_id:
    st.warning("🏠 사이드바의 **홈** 에서 Run 을 먼저 선택하고 모델을 로드해 주세요.")
    st.stop()

st.caption(f"Run: `{run_id}` · Models: {', '.join(active_model_ids())}")

# ----- 입력: CSV 업로드 or 텍스트 -----
tab_csv, tab_text = st.tabs(["📄 CSV 업로드", "📝 줄바꿈 텍스트"])

prompts: list[str] = []

with tab_csv:
    up = st.file_uploader(
        "CSV 파일 (`prompt` 컬럼 필수)",
        type=["csv"],
        help="UTF-8 인코딩. 헤더 `prompt` 가 있거나, 헤더 없이 첫 컬럼을 prompt 로 사용.",
    )
    if up is not None:
        try:
            # 헤더 자동 감지
            raw = up.getvalue().decode("utf-8", errors="replace")
            df_try = pd.read_csv(io.StringIO(raw))
            if "prompt" in df_try.columns:
                prompts = df_try["prompt"].dropna().astype(str).tolist()
            else:
                # 헤더 없는 파일로 재파싱
                df_try = pd.read_csv(io.StringIO(raw), header=None)
                prompts = df_try.iloc[:, 0].dropna().astype(str).tolist()
            st.success(f"✅ {len(prompts)}개 prompt 로드됨")
            with st.expander(f"미리보기 (최대 10개)"):
                st.dataframe(
                    pd.DataFrame({"prompt": prompts[:10]}),
                    hide_index=True, use_container_width=True,
                )
        except Exception as e:  # noqa: BLE001
            st.error(f"CSV 파싱 실패: {e}")

with tab_text:
    text_input = st.text_area(
        "Prompt 들 (줄바꿈으로 구분, 빈 줄 무시)",
        height=240,
        placeholder="The capital of France is\nIn a galaxy far far away\n= History =\n",
    )
    text_prompts = [p.strip() for p in text_input.split("\n") if p.strip()]
    if text_prompts and not prompts:
        prompts = text_prompts
        st.caption(f"{len(prompts)}개 prompt 감지됨")

st.divider()

# ----- 파라미터 -----
with st.expander("⚙️ 생성 파라미터", expanded=False):
    preset = st.selectbox("Preset", list(PARAM_PRESETS.keys()))
    pv = PARAM_PRESETS[preset]
    c1, c2, c3 = st.columns(3)
    with c1:
        max_new_tokens = st.slider("max_new_tokens", 16, 512, 80, step=16)
        do_sample = st.toggle("do_sample", value=pv["do_sample"])
    with c2:
        temperature = st.slider("temperature", 0.1, 2.0, float(pv["temperature"]),
                                 step=0.05, disabled=not do_sample)
        top_p = st.slider("top_p", 0.0, 1.0, float(pv["top_p"]), step=0.05,
                           disabled=not do_sample)
    with c3:
        top_k = st.slider("top_k", 0, 200, int(pv["top_k"]), disabled=not do_sample)
        repetition_penalty = st.slider("repetition_penalty", 1.0, 2.0, 1.1, step=0.05)
    seed = st.number_input("seed", value=42, step=1)

params = {
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "repetition_penalty": repetition_penalty,
    "do_sample": do_sample,
    "seed": int(seed),
}

# ----- 실행 -----
st.markdown(f"**실행 대상**: `{len(prompts)}` prompt × `{len(active_model_ids())}` 모델 = "
             f"**{len(prompts) * len(active_model_ids())}** 생성")

run_btn = st.button(
    "배치 생성",
    type="primary",
    disabled=len(prompts) == 0,
    use_container_width=True,
)

if run_btn:
    if len(prompts) > 200:
        st.error(f"prompt 수가 너무 많습니다 ({len(prompts)} > 200)")
        st.stop()

    progress = st.progress(0.0, text=f"0 / {len(prompts)}")
    result_items: list[dict] = []

    # 한 번의 API 호출은 서버에서 내부 루프 → 진행률 표시 안 됨.
    # UX 를 위해 client 측에서 prompt 하나씩 호출.
    for i, p in enumerate(prompts):
        try:
            resp = api_client.generate_batch(
                run_id=run_id,
                model_ids=active_model_ids(),
                prompts=[p],
                params=params,
            )
            result_items.extend(resp["items"])
        except Exception as e:  # noqa: BLE001
            st.error(f"prompt #{i+1} 실패: {e}")
            break
        progress.progress((i + 1) / len(prompts), text=f"{i+1} / {len(prompts)}")

    st.session_state["batch_results"] = {
        "run_id": run_id,
        "params": params,
        "items": result_items,
        "generated_at": datetime.now().isoformat(),
    }
    st.success(f"완료: {len(result_items)}건")

# ----- 결과 -----
results = st.session_state.get("batch_results")
if not results or not results.get("items"):
    st.stop()

st.divider()
st.markdown(f"### 📊 결과 ({len(results['items'])}건)")

# 평탄화 DataFrame
flat_rows = []
for item in results["items"]:
    row = {"prompt": item["prompt"]}
    for mid, out in item["outputs"].items():
        display = MODEL_DISPLAY.get(mid, mid)
        row[f"{display} - text"] = out.get("text", "")
        row[f"{display} - tok/s"] = out.get("tokens_per_sec", 0.0)
        row[f"{display} - ms"] = out.get("elapsed_ms", 0.0)
    flat_rows.append(row)

df = pd.DataFrame(flat_rows)

# 요약 통계
text_cols = [c for c in df.columns if c.endswith("- tok/s")]
if text_cols:
    st.markdown("**모델별 평균 tokens/sec**")
    summary = df[text_cols].mean().round(1).to_frame("평균 tok/s")
    st.dataframe(summary, use_container_width=True)

# 테이블
st.dataframe(df, hide_index=True, use_container_width=True, height=400)

# 다운로드
d1, d2 = st.columns(2)
with d1:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 CSV 다운로드",
        data=csv_bytes,
        file_name=f"batch_{run_id}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    jsonl_str = "\n".join(json.dumps(it, ensure_ascii=False) for it in results["items"])
    st.download_button(
        "💾 JSONL 다운로드",
        data=jsonl_str.encode("utf-8"),
        file_name=f"batch_{run_id}.jsonl",
        mime="application/x-ndjson",
        use_container_width=True,
    )

# 개별 prompt 자세히 보기
with st.expander("🔍 개별 prompt 자세히 보기"):
    idx = st.number_input(
        "index", min_value=0, max_value=len(results["items"]) - 1, value=0, step=1,
    )
    item = results["items"][int(idx)]
    st.markdown(f"**Prompt**: `{item['prompt']}`")
    cols = st.columns(len(item["outputs"]))
    for col, (mid, out) in zip(cols, item["outputs"].items()):
        with col:
            st.markdown(f"**{MODEL_DISPLAY.get(mid, mid)}**")
            st.caption(
                f"{out.get('num_tokens', 0)} tok · "
                f"{out.get('elapsed_ms', 0):.0f} ms · "
                f"{out.get('tokens_per_sec', 0):.1f} tok/s"
            )
            st.text_area(
                "text", value=out.get("text", ""), height=200,
                label_visibility="collapsed", disabled=True,
                key=f"batch_view_{idx}_{mid}",
            )
