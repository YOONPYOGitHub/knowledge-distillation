"""Storage 추상화 — 로컬 파일시스템 / Azure Blob 듀얼 백엔드.

모든 경로(rel_path)는 results/ 루트 기준의 POSIX 경로 문자열을 사용.
예: "logs/20260410_041631/summary.json", "checkpoints/{run_id}/student_kd_best.pt"
"""

from __future__ import annotations

import os
from typing import Optional

from .base import ResultsStorage

_storage: Optional[ResultsStorage] = None


def get_storage() -> ResultsStorage:
    """싱글턴 storage 인스턴스 반환. STORAGE_BACKEND 환경변수로 선택."""
    global _storage
    if _storage is not None:
        return _storage

    backend = os.getenv("STORAGE_BACKEND", "local").lower()
    if backend == "local":
        from .local import LocalStorage

        _storage = LocalStorage()
    elif backend == "blob":
        from .blob import BlobStorage

        _storage = BlobStorage()
    else:
        raise ValueError(f"알 수 없는 STORAGE_BACKEND: {backend}")

    return _storage


__all__ = ["ResultsStorage", "get_storage"]
