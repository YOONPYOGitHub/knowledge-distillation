"""로컬 파일시스템 기반 storage 구현 (results/ 디렉토리)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from ui.backend.config import RESULTS_DIR

from .base import ResultsStorage


class LocalStorage(ResultsStorage):
    def __init__(self, root: Path = RESULTS_DIR):
        self._root = root

    def _abs(self, rel_path: str) -> Path:
        return self._root / rel_path

    # ---- 메타 ----------------------------------------------------------

    def list_run_ids(self) -> list[str]:
        logs_dir = self._root / "logs"
        if not logs_dir.exists():
            return []
        runs = [
            entry.name
            for entry in logs_dir.iterdir()
            if entry.is_dir() and (entry / "summary.json").exists()
        ]
        runs.sort(reverse=True)
        return runs

    def exists(self, rel_path: str) -> bool:
        return self._abs(rel_path).exists()

    def list_files(self, rel_dir: str, suffix: str = "") -> list[dict]:
        d = self._abs(rel_dir)
        if not d.exists():
            return []
        items = []
        for p in sorted(d.iterdir()):
            if not p.is_file():
                continue
            if suffix and not p.name.endswith(suffix):
                continue
            stat = p.stat()
            items.append({
                "filename": p.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime),
            })
        return items

    # ---- 읽기 ----------------------------------------------------------

    def read_json(self, rel_path: str) -> Optional[dict | list]:
        p = self._abs(rel_path)
        if not p.exists():
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def read_text(self, rel_path: str) -> Optional[str]:
        p = self._abs(rel_path)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8")

    def read_bytes(self, rel_path: str) -> Optional[bytes]:
        p = self._abs(rel_path)
        if not p.exists():
            return None
        return p.read_bytes()

    # ---- 쓰기 ----------------------------------------------------------

    def write_text(self, rel_path: str, content: str) -> None:
        p = self._abs(rel_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    # ---- 바이너리 / URL -----------------------------------------------

    def local_path(self, rel_path: str) -> Path:
        p = self._abs(rel_path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

    def public_url(self, rel_path: str, expires_seconds: int = 3600) -> Optional[str]:
        # 로컬은 백엔드가 직접 서빙 → None
        return None
