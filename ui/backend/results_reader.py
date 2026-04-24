"""results/logs/{run_id}/summary.json 을 단일 소스로 스캔/파싱"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ui.backend.config import CHECKPOINTS_DIR, LOGS_DIR


# summary.json 의 "Teacher (FT)" 등을 내부 모델 ID 로 매핑
_RESULTS_KEY_MAP = {
    "Teacher (FT)": "teacher_ft",
    "Teacher": "teacher_ft",  # teacher_ft 가 없는 경우도 teacher 취급
    "Student (KD)": "student_kd",
    "Student (FT)": "student_ft",
    "Student (Base)": "student_base",
}


def list_run_ids() -> list[str]:
    """logs/ 디렉토리에서 summary.json 이 있는 run_id 목록 반환 (최신순)"""
    if not LOGS_DIR.exists():
        return []
    runs = []
    for entry in LOGS_DIR.iterdir():
        if entry.is_dir() and (entry / "summary.json").exists():
            runs.append(entry.name)
    # 이름(=타임스탬프) 기준 내림차순
    runs.sort(reverse=True)
    return runs


def load_summary(run_id: str) -> Optional[dict]:
    path = LOGS_DIR / run_id / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def checkpoint_path(run_id: str, model_id: str) -> Path:
    """model_id → checkpoint 파일 경로 (실제 존재 여부는 별도 확인)"""
    filename_map = {
        "teacher_ft": "teacher_ft_best.pt",
        "student_kd": "student_kd_best.pt",
        "student_ft": "student_ft_best.pt",
    }
    if model_id not in filename_map:
        raise ValueError(f"checkpoint 를 사용하지 않는 모델: {model_id}")
    return CHECKPOINTS_DIR / run_id / filename_map[model_id]


def get_run_info(run_id: str) -> Optional[dict]:
    """API 응답용 RunInfo 형태의 dict 반환 (summary + 체크포인트 존재 여부)"""
    summary = load_summary(run_id)
    if summary is None:
        return None

    config = summary.get("config", {})
    results_raw = summary.get("results", {})

    # 내부 ID로 변환
    results = {}
    for key, val in results_raw.items():
        mid = _RESULTS_KEY_MAP.get(key)
        if mid:
            results[mid] = {
                "ppl": val.get("ppl"),
                "tokens_per_sec": val.get("tokens_per_sec"),
            }

    # 체크포인트 존재 확인
    ckpt_dir = CHECKPOINTS_DIR / run_id
    checkpoints = {
        "teacher_ft": (ckpt_dir / "teacher_ft_best.pt").exists(),
        "student_kd": (ckpt_dir / "student_kd_best.pt").exists(),
        "student_ft": (ckpt_dir / "student_ft_best.pt").exists(),
        "student_base": True,  # 항상 pretrained 사용 가능
    }

    # created_at: run_id 가 YYYYMMDD_HHMMSS 형식이면 파싱
    created_at = None
    try:
        dt = datetime.strptime(run_id[:15], "%Y%m%d_%H%M%S")
        created_at = dt.isoformat()
    except (ValueError, IndexError):
        pass

    return {
        "run_id": run_id,
        "created_at": created_at,
        "teacher_model": config.get("teacher_model", "gpt2"),
        "student_model": config.get("student_model", "distilgpt2"),
        "checkpoints": checkpoints,
        "hyperparams": {
            k: config.get(k)
            for k in ("temperature", "alpha", "epochs", "learning_rate",
                      "max_seq_length", "batch_size")
            if k in config
        },
        "results": results,
    }


def list_run_infos() -> list[dict]:
    return [info for rid in list_run_ids() if (info := get_run_info(rid)) is not None]


def latest_run_id() -> Optional[str]:
    ids = list_run_ids()
    return ids[0] if ids else None
