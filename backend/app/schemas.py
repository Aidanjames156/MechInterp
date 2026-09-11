"""Pydantic request / response models. Mirrors the data model in the architecture sketch."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

Pooling = Literal["mean", "last"]


def _clean_prompts(prompts: list[str]) -> list[str]:
    cleaned = [p.strip() for p in prompts if p and p.strip()]
    if not cleaned:
        raise ValueError("at least one non-empty prompt is required")
    return cleaned


def _non_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("prompt must not be blank")
    return v


# ------------------------------------------------------------------ health
class Health(BaseModel):
    status: Literal["ok", "loading", "error"]
    model_loaded: bool
    model_name: str
    device: str
    n_layers: int | None = None
    d_model: int | None = None
    error: str | None = None


# ------------------------------------------------------------- prompt sets
class PromptSet(BaseModel):
    id: str
    name: str
    positive_prompts: list[str]
    negative_prompts: list[str]
    created_at: float


# ------------------------------------------------------------------ probes
class ExtractProbeRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    positive_prompts: list[str] = Field(min_length=1)
    negative_prompts: list[str] = Field(min_length=1)
    layer: int = Field(default=6, ge=0)
    pooling: Pooling = "mean"
    prompt_set_name: str | None = None

    _clean_pos = field_validator("positive_prompts")(_clean_prompts)
    _clean_neg = field_validator("negative_prompts")(_clean_prompts)


class ProbeSummary(BaseModel):
    id: str
    name: str
    prompt_set_id: str
    layer: int
    metadata: dict[str, Any]
    created_at: float


class Probe(ProbeSummary):
    vector: list[float]


# ----------------------------------------------------------------- measure
class MeasureRequest(BaseModel):
    prompt: str = Field(min_length=1)
    probe_ids: list[str] | None = None  # None = every saved probe
    pooling: Pooling = "mean"

    _non_blank = field_validator("prompt")(_non_blank)


class ProbeMeasurement(BaseModel):
    probe_id: str
    probe_name: str
    layer: int
    score: float
    cosine: float
    normalized: float  # 0 = like the training negatives, 1 = like the training positives
    pos_mean_projection: float
    neg_mean_projection: float
    per_layer: list[float]
    per_token: list[float]


class MeasureResponse(BaseModel):
    prompt: str
    tokens: list[str]
    results: list[ProbeMeasurement]


# ------------------------------------------------------------------- steer
class SteerRequest(BaseModel):
    probe_id: str
    prompt: str = Field(min_length=1)
    coefficient: float = Field(default=1.0, ge=-20, le=20)
    max_new_tokens: int = Field(default=40, ge=1, le=200)
    temperature: float = Field(default=0.7, ge=0, le=2)
    seed: int = 0

    _non_blank = field_validator("prompt")(_non_blank)


class LayerProjection(BaseModel):
    layer: int
    baseline: float
    intervened: float


class InterventionRun(BaseModel):
    id: str
    probe_id: str
    prompt: str
    mode: Literal["measure", "suppress", "boost"]
    coefficient: float
    output_baseline: str
    output_intervened: str
    activation_trace: list[LayerProjection]
    params: dict[str, Any]
    created_at: float
