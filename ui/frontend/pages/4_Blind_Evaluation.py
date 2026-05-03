"""Blind Evaluation — 모델 정체를 가린 상태에서 선호도/평가 수집

각 생성마다 model_ids 를 랜덤 셔플하여 Model A/B/C (또는 A/B/C/D) 로 표시.
사용자는 누가 누구인지 모르는 상태에서 점수와 선호를 매김.
저장 시 blind_mapping 도 함께 기록되어 나중에 집계 가능.
"""

from __future__ import annotations

import random
import string
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from ui.frontend.utils import api_client
from ui.frontend.utils.prompts import PARAM_PRESETS, PROMPT_TEMPLATES
from ui.frontend.utils.state import MODEL_DISPLAY, active_model_ids, init_session

st.set_page_config(page_title="Blind Evaluation", page_icon="🕶️", layout="wide")
init_session()

st.title("🕶️ Blind Evaluation")
st.caption("모델 이름을 가린 채로 생성 결과를 평가합니다. 세션 저장 후에만 정답을 확인할 수 있습니다.")

run_id = st.session_state.get("selected_run_id")
if not run_id:
    st.warning("🏠 사이드바의 **홈** 에서 Run 을 먼저 선택하고 모델을 로드해 주세요.")
    st.stop()

st.caption(f"Run: `{run_id}` · Models: {len(active_model_ids())}개")

# ----- Prompt + Params -----
col_prompt, col_params = st.columns([2, 1])

with col_prompt:
    def _on_blind_template():
        tpl = st.session_state["blind_prompt_template"]
        txt = PROMPT_TEMPLATES[tpl]
        if txt:
            st.session_state["blind_prompt_input"] = txt

    st.selectbox(
        "프롬프트 템플릿",
        list(PROMPT_TEMPLATES.keys()),
        index=0,
        key="blind_prompt_template",
        on_change=_on_blind_template,
    )
    prompt = st.text_area("Prompt", height=140, key="blind_prompt_input")

with col_params:
    st.markdown("**생성 파라미터**")
    preset = st.selectbox("Preset", list(PARAM_PRESETS.keys()), key="blind_preset")
    preset_vals = PARAM_PRESETS[preset]
    max_new_tokens = st.slider("max_new_tokens", 16, 512, 100, step=16, key="blind_mnt")
    do_sample = st.toggle("do_sample", value=preset_vals["do_sample"], key="blind_sample")
    temperature = st.slider("temperature", 0.1, 2.0, float(preset_vals["temperature"]),
                            step=0.05, disabled=not do_sample, key="blind_temp")
    top_p = st.slider("top_p", 0.0, 1.0, float(preset_vals["top_p"]),
                      step=0.05, disabled=not do_sample, key="blind_top_p")
    top_k = st.slider("top_k", 0, 200, int(preset_vals["top_k"]),
                      disabled=not do_sample, key="blind_top_k")
    repetition_penalty = st.slider("repetition_penalty", 1.0, 2.0, 1.1, step=0.05,
                                    key="blind_rep")
    seed = st.number_input("seed", value=42, step=1, key="blind_seed")

params = {
    "max_new_tokens": max_new_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "repetition_penalty": repetition_penalty,
    "do_sample": do_sample,
    "seed": int(seed),
}

run_btn = st.button("🎲 블라인드 생성 (셔플)", type="primary", use_container_width=True)
st.divider()

# ----- 생성 -----
if run_btn:
    if not prompt.strip():
        st.error("Prompt 가 비어 있습니다.")
        st.stop()

    model_ids = active_model_ids()
    with st.spinner("생성 중..."):
        try:
            resp = api_client.generate(
                run_id=run_id, model_ids=model_ids,
                prompt=prompt, params=params,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"생성 실패: {e}")
            st.stop()

    # 셔플: 실제 model_id 순서를 랜덤하게 A/B/C/D 로 매핑
    shuffled = list(resp["outputs"].keys())
    random.shuffle(shuffled)
    labels = list(string.ascii_uppercase[: len(shuffled)])  # A, B, C, ...
    blind_map = dict(zip(labels, shuffled))  # {A: "student_kd", B: "teacher_ft", ...}

    st.session_state["blind_resp"] = resp
    st.session_state["blind_mapping"] = blind_map
    st.session_state["blind_revealed"] = False
    st.session_state["blind_ratings"] = {}
    st.session_state["blind_preference"] = "tie"
    st.session_state["blind_comment"] = ""

resp = st.session_state.get("blind_resp")
blind_map = st.session_state.get("blind_mapping")

if not resp or not blind_map:
    st.info("블라인드 생성 버튼을 눌러 시작하세요.")
    st.stop()

revealed = st.session_state.get("blind_revealed", False)
labels = list(blind_map.keys())

# ----- 출력 (A/B/C) -----
cols = st.columns(len(labels))

LIKERT_FIELDS = [
    ("fluency", "유창성"),
    ("coherence", "일관성"),
    ("factuality", "사실성"),
    ("creativity", "창의성"),
]

st.session_state.setdefault("blind_ratings", {})

for col, label in zip(cols, labels):
    real_mid = blind_map[label]
    out = resp["outputs"][real_mid]
    with col:
        header = f"### Model {label}"
        if revealed:
            header += f"  _( **{MODEL_DISPLAY.get(real_mid, real_mid)}** )_"
        st.markdown(header)

        if out.get("error"):
            st.error(out["error"])
            continue

        c1, c2 = st.columns(2)
        c1.metric("latency", f"{out['elapsed_ms']:.0f} ms")
        c2.metric("tokens/s", f"{out['tokens_per_sec']:.1f}")
        st.caption(f"new tokens: {out['num_tokens']}")

        st.markdown("**출력:**")
        st.text_area(
            label=label,
            value=out["text"],
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )

        st.markdown("**평가 (1~5)**")
        model_ratings = st.session_state["blind_ratings"].setdefault(label, {})
        for key, name in LIKERT_FIELDS:
            model_ratings[key] = st.slider(
                name, 1, 5,
                value=model_ratings.get(key, 3),
                key=f"blind_rating_{label}_{key}",
            )

# ----- 선호도 + 코멘트 -----
st.divider()
st.markdown("### 🏆 종합 평가")
pref_col, comment_col = st.columns([1, 2])

with pref_col:
    pref_options = labels + ["tie"]
    pref_labels = [f"Model {l}" for l in labels] + ["차이 없음"]
    cur = st.session_state.get("blind_preference", "tie")
    try:
        default_idx = pref_options.index(cur)
    except ValueError:
        default_idx = len(pref_options) - 1
    idx = st.radio(
        "가장 마음에 드는 출력",
        options=range(len(pref_options)),
        format_func=lambda i: pref_labels[i],
        index=default_idx,
        key="blind_pref_radio",
    )
    st.session_state["blind_preference"] = pref_options[idx]

with comment_col:
    st.session_state["blind_comment"] = st.text_area(
        "코멘트 (선택)",
        value=st.session_state.get("blind_comment", ""),
        height=120,
        key="blind_comment_input",
    )

# ----- 저장 / 공개 -----
st.divider()
st.markdown("### 📥 세션 누적 & 공개")

bs1, bs2, bs3 = st.columns([1, 1, 1])

with bs1:
    if st.button("➕ 세션에 추가", type="primary", use_container_width=True):
        entry = {
            "run_id": run_id,
            "prompt": resp["prompt"],
            "params": params,
            "outputs": {
                label: {
                    "text": resp["outputs"][blind_map[label]]["text"],
                    "elapsed_ms": resp["outputs"][blind_map[label]]["elapsed_ms"],
                    "tokens_per_sec": resp["outputs"][blind_map[label]]["tokens_per_sec"],
                    "num_tokens": resp["outputs"][blind_map[label]]["num_tokens"],
                }
                for label in labels
            },
            "ratings": dict(st.session_state.get("blind_ratings", {})),
            "preference": st.session_state.get("blind_preference"),
            "comment": st.session_state.get("blind_comment", ""),
            "mode": "blind",
            "blind_mapping": blind_map,
        }
        st.session_state.setdefault("blind_history", []).append(entry)
        st.success(f"추가됨 (총 {len(st.session_state['blind_history'])}건)")

with bs2:
    n = len(st.session_state.get("blind_history", []))
    if st.button("🗑️ 세션 비우기", use_container_width=True, disabled=n == 0):
        st.session_state["blind_history"] = []
        st.rerun()

with bs3:
    if st.button(
        "👁️ 정답 공개" if not revealed else "🙈 가리기",
        use_container_width=True,
    ):
        st.session_state["blind_revealed"] = not revealed
        st.rerun()

history_count = len(st.session_state.get("blind_history", []))
st.metric("세션 누적", f"{history_count} 건")

if history_count > 0:
    import json as _json
    jsonl_str = "\n".join(
        _json.dumps(e, ensure_ascii=False)
        for e in st.session_state["blind_history"]
    )
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "💾 JSONL 다운로드",
            data=jsonl_str.encode("utf-8"),
            file_name=f"blind_{run_id}.jsonl",
            mime="application/x-ndjson",
            use_container_width=True,
        )
    with d2:
        if st.button("☁️ 서버 저장", use_container_width=True):
            try:
                r = api_client.save_feedback(
                    run_id=run_id,
                    entries=st.session_state["blind_history"],
                    filename_prefix="blind",
                )
                st.success(f"저장 완료: {r['path']} ({r['count']}건)")
            except Exception as e:  # noqa: BLE001
                st.error(f"저장 실패: {e}")

    with st.expander(f"최근 {min(5, history_count)}건 미리보기"):
        for i, e in enumerate(st.session_state["blind_history"][-5:], 1):
            st.markdown(f"**#{i}** — pref: `{e['preference']}` · "
                         f"mapping: {e['blind_mapping']}")
            st.caption(e["prompt"][:120])
