"""모델 로드 & 캐시 — run 단위로 LRU 관리"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ui.backend.config import MODEL_CACHE_SIZE, resolve_device
from ui.backend.results_reader import checkpoint_rel_path, get_run_info
from ui.backend.storage import get_storage

log = logging.getLogger(__name__)


@dataclass
class RunModels:
    """단일 run 의 로드된 모델 집합"""
    run_id: str
    teacher_model_name: str
    student_model_name: str
    tokenizer: "AutoTokenizer"
    models: dict[str, "AutoModelForCausalLM"] = field(default_factory=dict)

    def get(self, model_id: str):
        return self.models.get(model_id)

    def unload(self):
        self.models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelRegistry:
    """Run 단위 LRU 캐시."""

    def __init__(self, max_runs: int = MODEL_CACHE_SIZE):
        self._cache: OrderedDict[str, RunModels] = OrderedDict()
        self._tokenizer_cache: dict[str, "AutoTokenizer"] = {}
        self._max_runs = max_runs
        self._lock = Lock()
        self._device = resolve_device()

    @property
    def device(self) -> str:
        return self._device

    def _get_tokenizer(self, model_name: str) -> "AutoTokenizer":
        if model_name not in self._tokenizer_cache:
            tok = AutoTokenizer.from_pretrained(model_name)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tokenizer_cache[model_name] = tok
        return self._tokenizer_cache[model_name]

    def _load_single_model(
        self, model_name: str, checkpoint: Optional[str] = None
    ) -> "AutoModelForCausalLM":
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            log.info("  로드된 체크포인트: %s", checkpoint)
        model.to(self._device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def ensure_loaded(
        self, run_id: str, model_ids: list[str]
    ) -> RunModels:
        """요청된 model_ids 를 이 run 에 대해 로드. 이미 캐시돼 있으면 재사용."""
        with self._lock:
            info = get_run_info(run_id)
            if info is None:
                raise FileNotFoundError(f"run 을 찾을 수 없음: {run_id}")

            # LRU 갱신
            if run_id in self._cache:
                self._cache.move_to_end(run_id)
                run_models = self._cache[run_id]
            else:
                # 캐시 공간 확보
                while len(self._cache) >= self._max_runs:
                    old_id, old = self._cache.popitem(last=False)
                    log.info("LRU 제거: run_id=%s", old_id)
                    old.unload()

                tokenizer = self._get_tokenizer(info["student_model"])
                run_models = RunModels(
                    run_id=run_id,
                    teacher_model_name=info["teacher_model"],
                    student_model_name=info["student_model"],
                    tokenizer=tokenizer,
                )
                self._cache[run_id] = run_models

            # 개별 모델 로드
            for mid in model_ids:
                if mid in run_models.models:
                    continue

                storage = get_storage()
                if mid == "teacher_ft":
                    rel = checkpoint_rel_path(run_id, mid)
                    if not storage.exists(rel):
                        # FT 없으면 pretrained teacher 로 폴백
                        log.warning("teacher_ft 체크포인트 없음 → pretrained 사용")
                        model = self._load_single_model(run_models.teacher_model_name)
                    else:
                        model = self._load_single_model(
                            run_models.teacher_model_name, str(storage.local_path(rel))
                        )
                elif mid == "student_kd":
                    rel = checkpoint_rel_path(run_id, mid)
                    if not storage.exists(rel):
                        raise FileNotFoundError(f"체크포인트 없음: {rel}")
                    model = self._load_single_model(
                        run_models.student_model_name, str(storage.local_path(rel))
                    )
                elif mid == "student_ft":
                    rel = checkpoint_rel_path(run_id, mid)
                    if not storage.exists(rel):
                        raise FileNotFoundError(f"체크포인트 없음: {rel}")
                    model = self._load_single_model(
                        run_models.student_model_name, str(storage.local_path(rel))
                    )
                elif mid == "student_base":
                    model = self._load_single_model(run_models.student_model_name)
                else:
                    raise ValueError(f"알 수 없는 model_id: {mid}")

                run_models.models[mid] = model
                log.info("모델 로드 완료: run=%s model=%s", run_id, mid)

            return run_models

    def get(self, run_id: str) -> Optional[RunModels]:
        return self._cache.get(run_id)

    def memory_mb(self, run_models: RunModels) -> float:
        total_bytes = 0
        for m in run_models.models.values():
            total_bytes += sum(p.numel() * p.element_size() for p in m.parameters())
        return total_bytes / 1e6


# 싱글턴
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
