"""Azure Blob Storage 기반 storage 구현.

준비물 (의존성):
  - azure-storage-blob
  - azure-identity

환경변수:
  - AZURE_STORAGE_ACCOUNT     : 스토리지 계정 이름 (필수)
  - AZURE_BLOB_CONTAINER      : 컨테이너 이름 (기본: kd-results)
  - AZURE_BLOB_CACHE_DIR      : ckpt 다운로드 캐시 (기본: /tmp/kd-cache)
  - AZURE_BLOB_CACHE_MAX_GB   : 캐시 최대 용량 GB (기본: 4)

인증:
  - 기본은 DefaultAzureCredential (Container Apps 의 Managed Identity 자동 사용)
  - 로컬 테스트 시에는 az login 또는 AZURE_CLIENT_ID/SECRET/TENANT_ID 환경변수
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .base import ResultsStorage

log = logging.getLogger(__name__)


class BlobStorage(ResultsStorage):
    def __init__(self):
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.blob import BlobServiceClient
        except ImportError as e:
            raise ImportError(
                "BlobStorage 사용에는 azure-storage-blob, azure-identity 가 필요합니다. "
                "pip install azure-storage-blob azure-identity"
            ) from e

        account = os.environ["AZURE_STORAGE_ACCOUNT"]
        self._container_name = os.getenv("AZURE_BLOB_CONTAINER", "kd-results")
        self._account_url = f"https://{account}.blob.core.windows.net"

        self._credential = DefaultAzureCredential()
        self._service = BlobServiceClient(
            account_url=self._account_url, credential=self._credential
        )
        self._container = self._service.get_container_client(self._container_name)

        self._cache_dir = Path(
            os.getenv("AZURE_BLOB_CACHE_DIR", "/tmp/kd-cache")
        )
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_max_bytes = int(
            float(os.getenv("AZURE_BLOB_CACHE_MAX_GB", "4")) * 1024**3
        )
        self._cache_lock = threading.Lock()

        # SAS 발급용 user delegation key (1시간 캐시)
        self._udk = None
        self._udk_expiry = datetime.min.replace(tzinfo=timezone.utc)

    # ---- 메타 ----------------------------------------------------------

    def list_run_ids(self) -> list[str]:
        """logs/ 하위 하위 디렉터리 중 summary.json 이 있는 것."""
        run_ids: list[str] = []
        # delimiter='/' 로 walk: logs/{run_id}/summary.json 패턴
        for blob in self._container.list_blobs(name_starts_with="logs/"):
            name = blob.name  # ex) "logs/20260410_041631/summary.json"
            parts = name.split("/")
            if len(parts) == 3 and parts[2] == "summary.json":
                run_ids.append(parts[1])
        run_ids.sort(reverse=True)
        return run_ids

    def exists(self, rel_path: str) -> bool:
        return self._container.get_blob_client(rel_path).exists()

    def list_files(self, rel_dir: str, suffix: str = "") -> list[dict]:
        prefix = rel_dir.rstrip("/") + "/"
        items = []
        for blob in self._container.list_blobs(name_starts_with=prefix):
            tail = blob.name[len(prefix):]
            if "/" in tail:
                continue  # 하위 디렉터리는 제외
            if suffix and not tail.endswith(suffix):
                continue
            items.append({
                "filename": tail,
                "size_bytes": blob.size,
                "modified": blob.last_modified,
            })
        items.sort(key=lambda x: x["filename"])
        return items

    # ---- 읽기 ----------------------------------------------------------

    def read_bytes(self, rel_path: str) -> Optional[bytes]:
        client = self._container.get_blob_client(rel_path)
        if not client.exists():
            return None
        return client.download_blob().readall()

    def read_text(self, rel_path: str) -> Optional[str]:
        data = self.read_bytes(rel_path)
        return data.decode("utf-8") if data is not None else None

    def read_json(self, rel_path: str) -> Optional[dict | list]:
        text = self.read_text(rel_path)
        return json.loads(text) if text is not None else None

    # ---- 쓰기 ----------------------------------------------------------

    def write_text(self, rel_path: str, content: str) -> None:
        client = self._container.get_blob_client(rel_path)
        client.upload_blob(content.encode("utf-8"), overwrite=True)

    # ---- 바이너리 / URL -----------------------------------------------

    def local_path(self, rel_path: str) -> Path:
        """필요 시 Blob → 캐시 다운로드 후 로컬 경로 반환."""
        cached = self._cache_dir / rel_path
        if cached.exists():
            cached.touch(exist_ok=True)  # LRU mtime 갱신
            return cached

        with self._cache_lock:
            if cached.exists():
                return cached
            client = self._container.get_blob_client(rel_path)
            if not client.exists():
                raise FileNotFoundError(f"blob 없음: {rel_path}")

            cached.parent.mkdir(parents=True, exist_ok=True)
            tmp = cached.with_suffix(cached.suffix + ".part")
            log.info("Blob 다운로드: %s → %s", rel_path, cached)
            with open(tmp, "wb") as f:
                client.download_blob().readinto(f)
            tmp.rename(cached)
            self._evict_if_needed()
            return cached

    def public_url(self, rel_path: str, expires_seconds: int = 3600) -> Optional[str]:
        """User-delegation SAS URL 발급 (계정 키 불필요)."""
        try:
            from azure.storage.blob import BlobSasPermissions, generate_blob_sas
        except ImportError:
            return None

        if not self.exists(rel_path):
            return None

        udk = self._get_udk()
        sas = generate_blob_sas(
            account_name=self._service.account_name,
            container_name=self._container_name,
            blob_name=rel_path,
            user_delegation_key=udk,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(seconds=expires_seconds),
        )
        return f"{self._account_url}/{self._container_name}/{rel_path}?{sas}"

    # ---- 내부 헬퍼 -----------------------------------------------------

    def _get_udk(self):
        now = datetime.now(timezone.utc)
        if self._udk is None or now >= self._udk_expiry - timedelta(minutes=5):
            self._udk = self._service.get_user_delegation_key(
                key_start_time=now - timedelta(minutes=5),
                key_expiry_time=now + timedelta(hours=1),
            )
            self._udk_expiry = now + timedelta(hours=1)
        return self._udk

    def _evict_if_needed(self) -> None:
        """캐시 디렉터리가 한도 초과면 mtime 오래된 순으로 삭제."""
        files = []
        total = 0
        for p in self._cache_dir.rglob("*"):
            if p.is_file():
                size = p.stat().st_size
                files.append((p.stat().st_mtime, size, p))
                total += size
        if total <= self._cache_max_bytes:
            return
        files.sort()  # 오래된 것부터
        for _, size, p in files:
            if total <= self._cache_max_bytes:
                break
            try:
                p.unlink()
                total -= size
                log.info("캐시 evict: %s (%.1fMB)", p, size / 1e6)
            except OSError:
                pass
        # 빈 디렉터리 정리
        for p in sorted(self._cache_dir.rglob("*"), reverse=True):
            if p.is_dir() and not any(p.iterdir()):
                try:
                    p.rmdir()
                except OSError:
                    pass
