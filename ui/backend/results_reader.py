"""results/logs/{run_id}/summary.json 을 단일 소스로 스캔/파싱

storage 추상화를 통해 로컬/Blob 양쪽에서 동작.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ui.backend.storage import get_storage


# summary.json 의 "Teacher (FT)" 등을 내부 모델 ID 로 매핑
_RESULTS_KEY_MAP = {
    "Teacher (FT)": "teacher_ft",
    "Teacher": "teacher_ft",  # teacher_ft 가 없는 경우도 teacher 취급
    "Student (KD)": "student_kd",
    "Student (FT)": "student_ft",
    "Student (Base)": "student_base",
}

_CKPT_FILENAMES = {
    "teacher_ft": "teacher_ft_best.pt",
    "student_kd": "student_kd_best.pt",
    "student_ft": "student_ft_best.pt",
}


def list_run_ids() -> list[str]:
    """summary.json 이 있는 run_id 목록 반환 (최신순)."""
    return get_storage().list_run_ids()


def load_summary(run_id: str) -> Optional[dict]:
    return get_storage().read_json(f"logs/{run_id}/summary.json")


def checkpoint_rel_path(run_id: str, model_id: str) -> str:
    """체크포인트 storage 상대 경로."""
    if model_id not in _CKPT_FILENAMES:
        raise ValueError(f"checkpoint 를 사용하지 않는 모델: {model_id}")
    return f"checkpoints/{run_id}/{_CKPT_FILENAMES[model_id]}"


def checkpoint_exists(run_id: str, model_id: str) -> bool:
    return get_storage().exists(checkpoint_rel_path(run_id, model_id))


def get_run_info(run_id: str) -> Optional[dict]:
    """API 응답용 RunInfo 형태의 dict 반환 (summary + 체크포인트 존재 여부)."""
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
    storage = get_storage()
    checkpoints = {
        mid: storage.exists(f"checkpoints/{run_id}/{fname}")
        for mid, fname in _CKPT_FILENAMES.items()
    }
    checkpoints["student_base"] = True  # 항상 pretrained 사용 가능

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
