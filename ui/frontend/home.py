"""홈 — 대시보드 / 사용 가이드 / FAQ 탭 구성."""

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

tab_dash, tab_guide = st.tabs(["📊 대시보드", "📖 사용 가이드"])

# ╔════════════════════════════════════════════════════════════════╗
# ║ 탭 1: 대시보드                                                 ║
# ╚════════════════════════════════════════════════════════════════╝
with tab_dash:
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


# ╔════════════════════════════════════════════════════════════════╗
# ║ 탭 2: 사용 가이드                                              ║
# ╚════════════════════════════════════════════════════════════════╝
with tab_guide:
    st.markdown("### 🚀 5분만에 시작하기")
    st.markdown(
        """
        1. **Backend 상태 확인** — 페이지 우측 상단 `✅ Backend OK` 인지 확인
        2. **대시보드 탭** → Run 선택 → **모델 로드** 버튼 클릭
        3. **Compare Generate** 페이지로 이동해 prompt 입력 → 생성 비교
        4. **Blind Evaluation** 에서 어느 출력이 좋은지 라벨링
        5. **Aggregation** 에서 누적된 평가를 통계로 확인

        > 학습 결과(run)가 없다면 먼저 `uv run python main.py configs/exp08_alpha02_epoch8.yaml` 실행.
        """
    )

    st.divider()

    st.markdown("### 🧭 전체 워크플로우")
    st.markdown(
        """
        | 단계 | 페이지 | 목적 |
        |---|---|---|
        | 1 | 🏠 **Home** | Run 선택 + 모델 로드 (모든 페이지의 진입점) |
        | 2 | 🆚 **Compare Generate** | 단일 prompt 로 모델별 출력 즉시 비교 |
        | 3 | 📦 **Batch Prompts** | CSV/줄바꿈 텍스트로 여러 prompt 한꺼번에 생성 |
        | 4 | 🕶️ **Blind Evaluation** | 모델 라벨을 A/B/C로 가린 채 리커트 점수·선호 평가 |
        | 5 | 📈 **Aggregation** | 저장된 feedback JSONL 을 선호도/리커트 평균으로 집계 |
        | 6 | 📊 **Experiment Report** | Loss/PPL/속도 등 정량 지표, 하이퍼파라미터 스캔 |
        | 7 | 🔬 **Token Analysis** | 위치별 next-token 분포 + KL divergence |
        | 8 | 💾 **Checkpoint Browser** | 다른 run 으로 빠르게 전환·로드 |
        """
    )

    st.divider()

    # ── 페이지별 상세 ──────────────────────────────────────────────
    st.markdown("### 📄 페이지별 상세")

    with st.expander("🏠 Home (현재 페이지)", expanded=False):
        st.markdown(
            """
            **목적**: 모든 작업의 시작점. 어떤 run 의 어떤 모델을 쓸지 결정.

            **순서**
            1. `Run ID` 드롭다운에서 분석할 실험 선택
            2. `Student (Base) 포함` 토글 — 기본 3개(Teacher FT / KD / FT)에 Base 까지 추가할지
            3. **모델 로드** 버튼 — 백엔드 메모리에 모델을 미리 적재 (콜드 스타트 회피)

            **빠른 시작 영역**의 metric 으로 현재 상태(Run / 로드 모델 수 / Device)를 한눈에 확인.
            """
        )

    with st.expander("🆚 Compare Generate"):
        st.markdown(
            """
            **목적**: 같은 prompt 를 모델 3~4개에 동시에 던져 출력을 나란히 비교.

            **사용법**
            1. prompt 입력 (혹은 템플릿 선택)
            2. 생성 파라미터 조정: `max_new_tokens`, `temperature`, `top_p`
            3. **생성** 버튼 → 모델별 카드에 출력과 메트릭(latency, tok/s) 표시
            4. 각 카드 하단 **평가 (1~5)** 슬라이더로 유창성 / 일관성 / 사실성 / 창의성을 점수화
            5. 하단 **🏆 종합 평가**에서 가장 마음에 드는 출력 선택 + 코멘트 입력
            6. **➕ 세션에 추가** 로 누적 → **저장** 버튼으로 `results/qualitative/` 에 JSONL 기록 (추후 Aggregation 분석에 사용)
            """
        )

    with st.expander("📦 Batch Prompts"):
        st.markdown(
            """
            **목적**: 연구용 대량 샘플 수집. 여러 prompt × 모든 로드된 모델 → CSV/JSONL 일괄 다운로드.

            **입력 방식** (탭)
            - **📄 CSV 업로드**: `prompt` 컴럼이 있으면 해당 컴럼, 없으면 첫 컴럼을 prompt 로 사용 (UTF-8)
            - **📝 줄바꿈 텍스트**: 한 줄에 prompt 하나 (빈 줄 무시)

            **제약**: prompt 최대 200개. 생성 중 진행률 bar 로 증가 상태 표시.

            **결과**: 모델별 평균 tok/s 요약 + 평탄 DataFrame, **CSV** / **JSONL** 다운로드 버튼.
            """
        )

    with st.expander("🕶️ Blind Evaluation"):
        st.markdown(
            """
            **목적**: 모델명을 가린 상태로 사람이 출력 품질만 보고 평가.

            **사용법**
            1. prompt 입력 → 🎲 **블라인드 생성 (셔플)** → 실제 모델 순서가 랜덤으로 A/B/C로 매핑
            2. 각 Model A/B/C 카드 하단 **평가 (1~5)** 슬라이더로 유창성 / 일관성 / 사실성 / 창의성 점수화
            3. 하단 **🏆 종합 평가**에서 가장 마음에 드는 Model 선택 + 코멘트
            4. **➕ 세션에 추가** → **저장** 하면 `blind_mapping` 포함해 JSONL 로 기록
            5. **🔓 정체 공개** 버튼으로 각 라벨이 실제 어떤 모델이었는지 확인

            저장된 레코드는 **Aggregation** 에서 실제 model_id 로 역매핑되어 집계됨.
            """
        )

    with st.expander("📈 Aggregation"):
        st.markdown(
            """
            **목적**: 서버에 저장된 feedback JSONL 을 모아 정성 평가 결과를 집계.

            **필터**
            - 집계할 Run 멀티셀렉트 (여러 개 동시 선택)
            - 모드 필터: 전체 / `compare` / `blind` (블라인드는 `blind_mapping` 으로 자동 역매핑)

            **탭 구성**
            - 🏆 **선호도**: 가장 마음에 든 출력의 count / 비율 (bar 차트 + 표)
            - ⭐ **리커트 평균**: 모델별 4개 항목 평균 표 + 항목별 grouped bar + (항목 3개 이상) 레이더 차트
            - 📋 **원본 entries**: run/mode/prompt/preference/source 요약 + JSON 미리보기
            """
        )

    with st.expander("📊 Experiment Report"):
        st.markdown(
            """
            **목적**: 정량 지표 중심의 실험 보고 대시보드.

            **사용법**: 상단에서 비교할 run 들을 멀티셀렉트 (기본: 최신 3개) → 요약 테이블 + 아래 6개 탭.

            **탭 구성**
            - 📉 Loss 곡선: training/val loss
            - 🎯 PPL 비교: 모델별 perplexity
            - ⚡ 속도 & 메모리: tok/s, MB
            - 🔍 하이퍼파라미터 스캔: alpha/T 등 grid 결과
            - 🖼️ 기존 차트 (PNG): `compare.py` 로 생성된 PNG 그대로 임베드
            - 🆚 Run Diff: 두 run 의 하이퍼파라미터/지표 차이
            """
        )

    with st.expander("🔬 Token Analysis"):
        st.markdown(
            """
            **목적**: 모델이 **prompt 의 어느 위치**에서 무슨 토큰을 얼마나 확신하는지 분석 (teacher-forcing 방식).

            **파라미터**
            - **Teacher 기준**: KL divergence 의 P 분포로 쓸 모델 선택 (기본 `teacher_ft`)
            - **Top-K** (3~20): 각 위치에서 볼 후보 단어 개수
            - **분포 온도 T**: softmax 전 logits 을 T 로 나눔 (T=1이 원 분포)

            **결과**
            - 위치별 top-k 분포
            - Teacher 대비 각 student 모델의 KL divergence — 어디서 의견이 갈리는지
            - prompt 길이가 길면 속도가 느려질 수 있음
            """
        )

    with st.expander("💾 Checkpoint Browser"):
        st.markdown(
            """
            **목적**: 저장된 모든 run 을 카드 그리드로 탐색하고 빠르게 전환.

            **필터**
            - 검색 (run_id 일부 매칭)
            - **3종 체크포인트 모두 있는 run 만** 토글 (teacher_ft / student_kd / student_ft)

            **카드 내용**: 체크포인트 존재 배지(✅/❌) / 하이퍼파라미터 / 모델별 PPL · tok/s 요약

            **버튼 차이**
            - **현재 Run 으로 설정**: `selected_run_id` 만 갱신 (가볍움)
            - **모델 로드**: 위 동작 + 백엔드 메모리에 모델까지 적재 (즉시 생성 가능)
            """
        )
