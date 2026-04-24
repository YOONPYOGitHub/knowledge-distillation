# KD UI — 정성 & 정량 평가 플랫폼

> FastAPI (backend) + Streamlit (frontend) 기반 모델 비교/평가 웹앱.
> 자세한 설계는 [../docs/ui-development-plan.md](../docs/ui-development-plan.md) 참고.

---

## 설치

UI extras 설치:
```bash
uv sync --extra ui
```

---

## 실행

### 방법 1 — 한 번에 (권장)
```bash
make up
```

### 방법 2 — 분리 실행

터미널 1 (백엔드):
```bash
make backend
# → http://localhost:8000 (docs: /docs)
```

터미널 2 (프론트엔드):
```bash
make frontend
# → http://localhost:8501
```

---

## 환경변수

`ui/.env` (없으면 `.env.example` 복사):

```env
API_HOST=127.0.0.1
API_PORT=8000
UI_PORT=8501
API_BASE_URL=http://127.0.0.1:8000
DEFAULT_RUN_ID=            # 비어있으면 results/logs/ 중 최신 run 자동 선택
MODEL_CACHE_SIZE=2
DEVICE=auto                # auto | cuda | mps | cpu
```

---

## 폴더 구조

```
ui/
├── README.md
├── Makefile
├── .env.example
├── backend/                      # FastAPI
│   ├── app.py                    #   엔트리, lifespan
│   ├── schemas.py                #   Pydantic 요청/응답
│   ├── config.py                 #   UI 전용 설정
│   ├── model_registry.py         #   run_id → 체크포인트 캐시
│   ├── generator.py              #   동기/스트리밍 생성 공통
│   ├── token_analyzer.py         #   top-k + KL divergence
│   ├── results_reader.py         #   summary.json / history 파싱
│   └── routers/
│       ├── runs.py               #   GET /runs, /runs/{id}/*
│       ├── generate.py           #   POST /generate
│       ├── generate_stream.py    #   POST /generate/stream  (SSE)
│       ├── generate_batch.py     #   POST /generate/batch
│       ├── token_analysis.py     #   POST /token-analysis
│       ├── feedback.py           #   POST /feedback/save + entries
│       └── reports.py            #   figures 프록시
└── frontend/                     # Streamlit
    ├── app.py                    #   엔트리 · st.navigation 기반 멀티페이지 라우팅
    ├── home.py                   #   홈 페이지 (Run 선택 · 모델 로드 · 페이지 안내)
    ├── utils/
    │   ├── api_client.py
    │   ├── state.py
    │   └── prompts.py
    └── pages/
        ├── 1_Compare_Generate.py  # 3/4열 비교, 실시간 스트리밍 토글
        ├── 2_Experiment_Report.py # 6탭 (loss/PPL/속도/하이퍼/diff/figures)
        ├── 3_Checkpoint_Browser.py
        ├── 4_Blind_Evaluation.py  # A/B/C 블라인드 + 리커트 + JSONL 저장
        ├── 5_Aggregation.py       # 블라인드 집계 대시보드
        ├── 6_Token_Analysis.py    # top-k 분포 + Teacher‖Student KL
        └── 7_Batch_Prompts.py     # CSV / 텍스트 배치 생성 + CSV/JSONL 다운로드
```

---

## 페이지 요약

| # | 페이지 | 기능 |
|---|---|---|
| 1 | **Compare Generate** | Teacher(FT) / Student(KD) / Student(FT) 3열 비교, `Student(Base)` 토글, ⚡ **실시간 스트리밍** 토글 (SSE) |
| 2 | **Experiment Report** | 여러 run 겹쳐보기: Loss 곡선, 4-Way PPL, 속도/메모리, 하이퍼 스캔, Run Diff, 기존 figures |
| 3 | **Checkpoint Browser** | run별 체크포인트 존재/크기, 클릭으로 로드 |
| 4 | **Blind Evaluation** | A/B/C 셔플 + 4축 리커트 + 선호도 + 코멘트 → JSONL 저장 |
| 5 | **Aggregation** | 블라인드 JSONL 집계 — 모델별 선호 비율, 리커트 평균 |
| 6 | **Token Analysis** | 각 토큰 위치의 top-k 분포 + Teacher‖KD / Teacher‖FT **KL divergence** |
| 7 | **Batch Prompts** | CSV 업로드 / 멀티라인 텍스트 → 배치 생성, 진행률 바, CSV·JSONL 다운로드 |

샘플 배치 파일: [`../examples/sample_batch_prompts.csv`](../examples/sample_batch_prompts.csv) — 50 prompt × 6 카테고리 (factual/narrative/technical/wiki/dialogue/completion).

---

## API 문서

백엔드 실행 후 자동 생성:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

주요 엔드포인트:

| Method | Path | 설명 |
|---|---|---|
| GET  | `/runs` | 학습 결과 run 목록 |
| GET  | `/runs/{run_id}/history` | baseline / distill / teacher history 통합 |
| GET  | `/runs/{run_id}/evaluation` | 4-Way 평가 결과 |
| GET  | `/runs/{run_id}/figures/{name}` | PNG 이미지 프록시 |
| POST | `/models/load` | 특정 run의 모델 메모리 로드 |
| POST | `/generate` | 동기 다중 모델 생성 |
| POST | `/generate/stream` | SSE 스트리밍 (토큰 단위, 모델 병렬) |
| POST | `/generate/batch` | 다수 prompt × 다수 모델 배치 생성 (≤200 prompts) |
| POST | `/token-analysis` | top-k 확률 + KL divergence |
| POST | `/feedback/save` | 정성 평가 저장 |
| GET  | `/feedback/entries` | 저장된 평가 조회 |

---

## 마일스톤

- **v0.1** MVP — 3열 비교 생성, Base 토글, 파라미터 패널 ✅
- **v0.2** 정성 평가 (리커트/선호도/JSONL) ✅
- **v0.3** Experiment Report + Checkpoint Browser ✅
- **v0.4** Blind Evaluation + 집계 ✅
- **v0.5** Token Analysis (top-k + KL divergence) ✅
- **v0.6** 실시간 스트리밍 + Batch Prompts (CSV) ✅ **← 현재**
