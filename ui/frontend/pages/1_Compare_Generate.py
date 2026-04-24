"""Compare Generate — 3(또는 4)열 비교 생성"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.frontend.utils import api_client
from ui.frontend.utils.prompts import PARAM_PRESETS, PROMPT_TEMPLATES
from ui.frontend.utils.state import MODEL_DISPLAY, active_model_ids, init_session

st.set_page_config(page_title="Compare Generate", page_icon="🆚", layout="wide")
init_session()

st.title("🆚 Compare Generate")

run_id = st.session_state.get("selected_run_id")
if not run_id:
    st.warning("🏠 사이드바의 **홈** 에서 Run 을 먼저 선택하고 모델을 로드해 주세요.")
    st.stop()

st.caption(f"Run: `{run_id}` · Models: {', '.join(active_model_ids())}")

# --- Prompt 입력 영역 ---
col_prompt, col_params = st.columns([2, 1])

with col_prompt:
    # 템플릿 선택이 바뀌면 text_area 의 session_state 값을 직접 갱신
    def _on_template_change():
        tpl = st.session_state["prompt_template"]
        txt = PROMPT_TEMPLATES[tpl]
        if txt:  # "(직접 입력)" 은 빈 문자열이라 기존 값 유지
            st.session_state["prompt_input"] = txt

    st.selectbox(
        "프롬프트 템플릿",
        list(PROMPT_TEMPLATES.keys()),
        index=0,
        key="prompt_template",
        on_change=_on_template_change,
    )
    prompt = st.text_area(
        "Prompt",
        height=140,
        key="prompt_input",
    )

with col_params:
    st.markdown("**생성 파라미터**")
    preset = st.selectbox("Preset", list(PARAM_PRESETS.keys()), index=0)
    preset_vals = PARAM_PRESETS[preset]

    max_new_tokens = st.slider("max_new_tokens", 16, 512, 100, step=16)
    do_sample = st.toggle("do_sample", value=preset_vals["do_sample"])
    temperature = st.slider(
        "temperature", 0.1, 2.0, float(preset_vals["temperature"]), step=0.05,
        disabled=not do_sample,
    )
    top_p = st.slider(
        "top_p", 0.0, 1.0, float(preset_vals["top_p"]), step=0.05,
        disabled=not do_sample,
    )
    top_k = st.slider(
        "top_k", 0, 200, int(preset_vals["top_k"]),
        disabled=not do_sample,
    )
    repetition_penalty = st.slider(
        "repetition_penalty", 1.0, 2.0, 1.1, step=0.05
    )
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

btn_col1, btn_col2 = st.columns([3, 2])
with btn_col1:
    run_btn = st.button("생성", type="primary", use_container_width=True)
with btn_col2:
    use_stream = st.toggle(
        "실시간 스트리밍",
        value=True,
        help="토큰 단위로 즉시 화면에 표시. 끄면 완료 후 한 번에 표시.",
    )
st.divider()

# --- 결과 출력 ---
if run_btn:
    if not prompt.strip():
        st.error("Prompt 가 비어 있습니다.")
        st.stop()

    active_ids = active_model_ids()

    if use_stream:
        # 스트리밍: 열 placeholder 를 먼저 그리고 토큰이 도착할 때마다 업데이트
        stream_cols = st.columns(len(active_ids))
        text_holders: dict[str, "st.delta_generator.DeltaGenerator"] = {}
        status_holders: dict[str, "st.delta_generator.DeltaGenerator"] = {}
        accum: dict[str, str] = {mid: "" for mid in active_ids}

        from ui.frontend.utils.state import MODEL_DISPLAY as _MD
        for col, mid in zip(stream_cols, active_ids):
            with col:
                st.markdown(f"### {_MD.get(mid, mid)}")
                status_holders[mid] = st.empty()
                text_holders[mid] = st.empty()
                status_holders[mid].caption("⏳ 대기 중...")

        final_stats: dict[str, dict] = {}

        try:
            for evt in api_client.generate_stream(
                run_id=run_id, model_ids=active_ids,
                prompt=prompt, params=params,
            ):
                t = evt.get("type")
                mid = evt.get("model_id")
                if t == "start":
                    status_holders[mid].caption(
                        f"🟡 생성 중... (prompt {evt['prompt_tokens']} 토큰)"
                    )
                elif t == "token":
                    accum[mid] += evt["text"]
                    text_holders[mid].markdown(
                        f"```\n{accum[mid]}▌\n```"
                    )
                elif t == "done":
                    final_stats[mid] = evt
                    status_holders[mid].caption(
                        f"✅ {evt['num_tokens']} 토큰 · "
                        f"{evt['elapsed_ms']:.0f} ms · "
                        f"{evt['tokens_per_sec']:.1f} tok/s"
                    )
                    text_holders[mid].markdown(f"```\n{accum[mid]}\n```")
                elif t == "error":
                    status_holders[mid].error(evt.get("error", "error"))
                elif t == "all_done":
                    break
        except Exception as e:  # noqa: BLE001
            st.error(f"스트리밍 실패: {e}")
            st.stop()

        # 평가용 resp 를 기존 스키마로 구성
        resp = {
            "run_id": run_id,
            "prompt": prompt,
            "outputs": {
                mid: {
                    "text": prompt + accum[mid],  # Compare 에선 full text 를 보여줌
                    "num_tokens": final_stats.get(mid, {}).get("num_tokens", 0),
                    "elapsed_ms": final_stats.get(mid, {}).get("elapsed_ms", 0.0),
                    "tokens_per_sec": final_stats.get(mid, {}).get("tokens_per_sec", 0.0),
                    "error": None,
                }
                for mid in active_ids
            },
        }
        st.session_state["last_generation"] = resp
        st.session_state["current_ratings"] = {}
        st.session_state["current_preference"] = "tie"
        st.session_state["current_comment"] = ""
        st.divider()
    else:
        with st.spinner("생성 중..."):
            try:
                resp = api_client.generate(
                    run_id=run_id,
                    model_ids=active_ids,
                    prompt=prompt,
                    params=params,
                )
                st.session_state["last_generation"] = resp
                st.session_state["current_ratings"] = {}
                st.session_state["current_preference"] = "tie"
                st.session_state["current_comment"] = ""
            except Exception as e:  # noqa: BLE001
                st.error(f"생성 실패: {e}")
                st.stop()

resp = st.session_state.get("last_generation")
if not resp:
    st.info("아직 생성 결과가 없습니다. 프롬프트 입력 후 **생성** 버튼을 눌러주세요.")
    st.stop()

model_ids = list(resp["outputs"].keys())
cols = st.columns(len(model_ids))

# 현재 평가 수집용 state
st.session_state.setdefault("current_ratings", {})

LIKERT_FIELDS = [
    ("fluency", "유창성"),
    ("coherence", "일관성"),
    ("factuality", "사실성"),
    ("creativity", "창의성"),
]

for col, mid in zip(cols, model_ids):
    out = resp["outputs"][mid]
    with col:
        st.markdown(f"### {MODEL_DISPLAY.get(mid, mid)}")
        if out.get("error"):
            st.error(out["error"])
            continue

        c1, c2 = st.columns(2)
        c1.metric("latency", f"{out['elapsed_ms']:.0f} ms")
        c2.metric("tokens/s", f"{out['tokens_per_sec']:.1f}")
        st.caption(f"new tokens: {out['num_tokens']}")

        st.markdown("**출력:**")
        # NOTE: key 를 지정하면 session_state 가 우선되어 재생성 시 갱신되지 않음.
        # 출력은 읽기 전용이므로 key 없이 value 만 사용.
        st.text_area(
            label=mid,
            value=out["text"],
            height=260,
            label_visibility="collapsed",
            disabled=True,
        )

        # --- 리커트 평가 ---
        st.markdown("**평가 (1~5)**")
        model_ratings = st.session_state["current_ratings"].setdefault(mid, {})
        for key, label in LIKERT_FIELDS:
            model_ratings[key] = st.slider(
                label,
                1, 5,
                value=model_ratings.get(key, 3),
                key=f"rating_{mid}_{key}",
            )

# --- 전체 선호도 & 코멘트 ---
st.divider()
st.markdown("### 🏆 종합 평가")
pref_col, comment_col = st.columns([1, 2])

with pref_col:
    pref_options = model_ids + ["tie"]
    pref_labels = [MODEL_DISPLAY.get(mid, mid) for mid in model_ids] + ["차이 없음"]
    current_pref = st.session_state.get("current_preference", "tie")
    try:
        default_idx = pref_options.index(current_pref)
    except ValueError:
        default_idx = len(pref_options) - 1
    choice_idx = st.radio(
        "가장 마음에 드는 출력",
        options=range(len(pref_options)),
        format_func=lambda i: pref_labels[i],
        index=default_idx,
        key="pref_radio",
    )
    st.session_state["current_preference"] = pref_options[choice_idx]

with comment_col:
    st.session_state["current_comment"] = st.text_area(
        "코멘트 (선택)",
        value=st.session_state.get("current_comment", ""),
        height=120,
        placeholder="예: Teacher 는 사실성이 높지만 반복이 많음, KD 는 자연스럽지만 주제 이탈…",
        key="comment_input",
    )

# --- 세션 누적 ---
st.divider()
st.markdown("### 📥 세션 누적 & 저장")

sess_col1, sess_col2, sess_col3 = st.columns([1, 1, 2])

with sess_col1:
    if st.button("➕ 세션에 추가", type="primary", use_container_width=True):
        entry = {
            "run_id": run_id,
            "prompt": resp["prompt"],
            "params": params,
            "outputs": {
                mid: {
                    "text": out["text"],
                    "elapsed_ms": out["elapsed_ms"],
                    "tokens_per_sec": out["tokens_per_sec"],
                    "num_tokens": out["num_tokens"],
                    "error": out.get("error"),
                }
                for mid, out in resp["outputs"].items()
            },
            "ratings": dict(st.session_state.get("current_ratings", {})),
            "preference": st.session_state.get("current_preference"),
            "comment": st.session_state.get("current_comment", ""),
            "mode": "compare",
        }
        st.session_state.setdefault("history", []).append(entry)
        st.success(f"추가됨 (총 {len(st.session_state['history'])}건)")

with sess_col2:
    history_count = len(st.session_state.get("history", []))
    if st.button(
        "🗑️ 세션 비우기",
        use_container_width=True,
        disabled=history_count == 0,
    ):
        st.session_state["history"] = []
        st.rerun()

with sess_col3:
    st.metric("세션 누적", f"{history_count} 건")

# 다운로드 + 서버 저장
if history_count > 0:
    import json as _json

    jsonl_str = "\n".join(
        _json.dumps(e, ensure_ascii=False) for e in st.session_state["history"]
    )
    st.download_button(
        "💾 JSONL 다운로드",
        data=jsonl_str.encode("utf-8"),
        file_name=f"session_{run_id}.jsonl",
        mime="application/x-ndjson",
    )

    if st.button("☁️ 서버에도 저장 (results/qualitative/)"):
        try:
            result = api_client.save_feedback(
                run_id=run_id,
                entries=st.session_state["history"],
                filename_prefix="session",
            )
            st.success(f"저장됨: `{result['path']}` ({result['count']}건)")
        except Exception as e:  # noqa: BLE001
            st.error(f"저장 실패: {e}")

# --- 세션 미리보기 ---
if history_count > 0:
    with st.expander(f"📋 세션 내역 미리보기 ({history_count}건)"):
        for i, entry in enumerate(st.session_state["history"][-5:], 1):
            idx = history_count - 5 + i if history_count > 5 else i
            st.markdown(f"**#{idx}** — `{entry['preference']}` · {entry['comment'][:60] or '(no comment)'}")
            st.caption(f"prompt: {entry['prompt'][:80]}...")
        if history_count > 5:
            st.caption(f"… (최근 5건만 표시. 전체 {history_count}건)")
