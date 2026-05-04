"""Storage 추상 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional


class ResultsStorage(ABC):
    """results/ 디렉토리에 대한 read/write 추상화.

    rel_path 는 results/ 루트 기준 POSIX 경로 ("logs/{run_id}/summary.json").
    """

    # ---- 메타/존재 확인 ------------------------------------------------

    @abstractmethod
    def list_run_ids(self) -> list[str]:
        """logs/ 하위에서 summary.json 이 있는 run_id 목록 (최신순)."""

    @abstractmethod
    def exists(self, rel_path: str) -> bool: ...

    @abstractmethod
    def list_files(self, rel_dir: str, suffix: str = "") -> list[dict]:
        """rel_dir 안의 파일 목록.

        반환 dict: {"filename": str, "size_bytes": int, "modified": datetime|None}
        """

    # ---- 텍스트/JSON 읽기 ----------------------------------------------

    @abstractmethod
    def read_json(self, rel_path: str) -> Optional[dict | list]:
        """JSON 파일 파싱. 없으면 None."""

    @abstractmethod
    def read_text(self, rel_path: str) -> Optional[str]:
        """텍스트 파일 전체. 없으면 None."""

    @abstractmethod
    def read_bytes(self, rel_path: str) -> Optional[bytes]: ...

    # ---- 텍스트 쓰기 (feedback 용) -------------------------------------

    @abstractmethod
    def write_text(self, rel_path: str, content: str) -> None:
        """전체 덮어쓰기. 부모 디렉토리가 없으면 자동 생성."""

    # ---- 바이너리 (체크포인트) -----------------------------------------

    @abstractmethod
    def local_path(self, rel_path: str) -> Path:
        """파일을 로컬 디스크에서 접근 가능한 Path 로 보장하고 반환.

        Local: results/ 안의 실제 경로를 그대로 반환.
        Blob: 캐시 디렉토리로 다운로드 후 그 경로 반환.

        파일이 없으면 FileNotFoundError.
        """

    # ---- 정적 자산 URL (figure 등) -------------------------------------

    @abstractmethod
    def public_url(self, rel_path: str, expires_seconds: int = 3600) -> Optional[str]:
        """Blob: SAS URL. Local: None (백엔드가 직접 서빙).

        반환값이 None 이면 호출 측에서 read_bytes/local_path 로 직접 응답.
        """
