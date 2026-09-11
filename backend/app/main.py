"""FastAPI app for the Mech Interp Sandbox.

Endpoints (v1, per the build plan):
    POST /extract_probe            build + save a probe from contrastive prompt sets
    POST /measure                  project a prompt onto saved probes
    POST /generate_with_steering   baseline vs. intervened generation
plus listing for probes, prompt sets and runs, and /health.

The model is loaded on a background thread at startup so the server answers
/health immediately; model endpoints return 503 until loading finishes.
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import schemas as S
from .db import Database
from .model import ModelService

MODEL_NAME = os.environ.get("MI_MODEL", "gpt2-small")
DB_PATH = os.environ.get(
    "MI_DB_PATH", str(Path(__file__).resolve().parent.parent / "data" / "sandbox.db")
)

model_service = ModelService(MODEL_NAME)
db = Database(DB_PATH)


def _load_in_background() -> None:
    try:
        model_service.load()
    except Exception:  # the error is recorded on the service and reported by /health
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_in_background, name="model-loader", daemon=True).start()
    yield
    db.close()


app = FastAPI(title="Mech Interp Sandbox", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_model() -> ModelService:
    if not model_service.loaded:
        if model_service.load_error:
            raise HTTPException(status_code=503, detail=f"model failed to load: {model_service.load_error}")
        raise HTTPException(status_code=503, detail="model is still loading; retry shortly")
    return model_service


def get_probe_or_404(probe_id: str) -> dict:
    probe = db.get_probe(probe_id)
    if probe is None:
        raise HTTPException(status_code=404, detail=f"probe {probe_id} not found")
    return probe


# ------------------------------------------------------------------ health
@app.get("/health", response_model=S.Health)
def health() -> S.Health:
    info = model_service.info()
    if model_service.loaded:
        status = "ok"
    elif model_service.load_error:
        status = "error"
    else:
        status = "loading"
    return S.Health(status=status, model_loaded=model_service.loaded, error=model_service.load_error, **info)


# ------------------------------------------------------------------ probes
@app.post("/extract_probe", response_model=S.Probe)
def extract_probe(req: S.ExtractProbeRequest) -> dict:
    svc = require_model()
    try:
        vector, metadata = svc.extract_probe(
            req.positive_prompts, req.negative_prompts, req.layer, req.pooling
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    prompt_set = db.create_prompt_set(
        req.prompt_set_name or req.name, req.positive_prompts, req.negative_prompts
    )
    return db.create_probe(req.name, prompt_set["id"], req.layer, vector, metadata)


@app.get("/probes", response_model=list[S.ProbeSummary])
def list_probes() -> list[dict]:
    return db.list_probes(include_vector=False)


@app.get("/probes/{probe_id}", response_model=S.Probe)
def get_probe(probe_id: str) -> dict:
    return get_probe_or_404(probe_id)


@app.delete("/probes/{probe_id}", status_code=204)
def delete_probe(probe_id: str) -> None:
    if not db.delete_probe(probe_id):
        raise HTTPException(status_code=404, detail=f"probe {probe_id} not found")


@app.get("/prompt_sets", response_model=list[S.PromptSet])
def list_prompt_sets() -> list[dict]:
    return db.list_prompt_sets()


@app.get("/prompt_sets/{prompt_set_id}", response_model=S.PromptSet)
def get_prompt_set(prompt_set_id: str) -> dict:
    ps = db.get_prompt_set(prompt_set_id)
    if ps is None:
        raise HTTPException(status_code=404, detail=f"prompt set {prompt_set_id} not found")
    return ps


# ----------------------------------------------------------------- measure
@app.post("/measure", response_model=S.MeasureResponse)
def measure(req: S.MeasureRequest) -> dict:
    svc = require_model()
    if req.probe_ids is None:
        probes = db.list_probes(include_vector=True)
    else:
        probes = [get_probe_or_404(pid) for pid in req.probe_ids]
    if not probes:
        return {"prompt": req.prompt, "tokens": [], "results": []}

    try:
        out = svc.measure(
            [{"vector": p["vector"], "layer": p["layer"]} for p in probes], req.prompt, req.pooling
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = []
    for probe, r in zip(probes, out["results"]):
        meta = probe["metadata"]
        pos_mean = meta.get("pos_mean_projection", 1.0)
        neg_mean = meta.get("neg_mean_projection", 0.0)
        spread = pos_mean - neg_mean
        normalized = (r["score"] - neg_mean) / spread if abs(spread) > 1e-12 else 0.0
        results.append({
            "probe_id": probe["id"],
            "probe_name": probe["name"],
            "layer": probe["layer"],
            "normalized": normalized,
            "pos_mean_projection": pos_mean,
            "neg_mean_projection": neg_mean,
            **r,
        })
    return {"prompt": req.prompt, "tokens": out["tokens"], "results": results}


# ------------------------------------------------------------------- steer
@app.post("/generate_with_steering", response_model=S.InterventionRun)
def generate_with_steering(req: S.SteerRequest) -> dict:
    svc = require_model()
    probe = get_probe_or_404(req.probe_id)
    try:
        out = svc.steer(
            vector=probe["vector"],
            layer=probe["layer"],
            raw_norm=probe["metadata"]["norm"],
            prompt=req.prompt,
            coefficient=req.coefficient,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            seed=req.seed,
            pooling=probe["metadata"].get("pooling", "mean"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    mode = "boost" if req.coefficient > 0 else "suppress" if req.coefficient < 0 else "measure"
    params = {
        "max_new_tokens": req.max_new_tokens,
        "temperature": req.temperature,
        "seed": req.seed,
        "layer": probe["layer"],
        "effective_scale": out["effective_scale"],
        "completion_baseline": out["completion_baseline"],
        "completion_intervened": out["completion_intervened"],
    }
    return db.create_run(
        probe_id=probe["id"],
        prompt=req.prompt,
        mode=mode,
        coefficient=req.coefficient,
        output_baseline=out["output_baseline"],
        output_intervened=out["output_intervened"],
        activation_trace=out["activation_trace"],
        params=params,
    )


@app.get("/runs", response_model=list[S.InterventionRun])
def list_runs(limit: int = 50) -> list[dict]:
    return db.list_runs(limit=max(1, min(limit, 500)))


@app.get("/runs/{run_id}", response_model=S.InterventionRun)
def get_run(run_id: str) -> dict:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return run
