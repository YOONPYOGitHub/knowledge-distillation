"""Pydantic schemas — 요청/응답 모델"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


ModelId = Literal["teacher_ft", "student_kd", "student_ft", "student_base"]


class RunCheckpoints(BaseModel):
    teacher_ft: bool
    student_kd: bool
    student_ft: bool
    student_base: bool = True  # pretrained 은 항상 사용 가능


class RunResults(BaseModel):
    ppl: Optional[float] = None
    tokens_per_sec: Optional[float] = None


class RunInfo(BaseModel):
    run_id: str
    created_at: Optional[str] = None
    teacher_model: str
    student_model: str
    checkpoints: RunCheckpoints
    hyperparams: dict
    results: dict[str, RunResults] = Field(default_factory=dict)


class LoadRequest(BaseModel):
    run_id: str
    models: list[ModelId] = Field(
        default_factory=lambda: ["teacher_ft", "student_kd", "student_ft"]
    )


class LoadResponse(BaseModel):
    run_id: str
    loaded: list[ModelId]
    device: str
    memory_mb: Optional[float] = None


class GenerationParams(BaseModel):
    max_new_tokens: int = Field(default=100, ge=1, le=1024)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    do_sample: bool = True
    seed: Optional[int] = 42


class GenerateRequest(BaseModel):
    run_id: str
    model_ids: list[ModelId]
    prompt: str
    params: GenerationParams = Field(default_factory=GenerationParams)


class ModelOutput(BaseModel):
    text: str
    num_tokens: int
    elapsed_ms: float
    tokens_per_sec: float
    error: Optional[str] = None


class GenerateResponse(BaseModel):
    run_id: str
    prompt: str
    outputs: dict[str, ModelOutput]
