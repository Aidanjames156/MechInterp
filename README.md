# Mech Interp Sandbox

A small lab tool for building and testing mechanistic-interpretability techniques
against GPT-2 small: linear probes, emotion / persona vectors, live measurement,
and activation steering. Implements v1 of the build plan in
[mech-interp-sandbox-architecture.md](mech-interp-sandbox-architecture.md).

- **Backend:** Python + FastAPI + [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), GPT-2 small (124M) held in memory, SQLite for probes and run logs.
- **Frontend:** React + Vite. Three panels: probe builder, live measurement, steering.

Everything runs on a laptop CPU. The first backend start downloads GPT-2 small (~500 MB) from Hugging Face.

## Setup

Requires Python 3.11+ and Node 20+.

```powershell
# backend
cd backend
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r requirements.txt

# frontend
cd ..\frontend
npm install
```

## Run

Two terminals:

```powershell
# terminal 1 - API on http://127.0.0.1:8000  (docs at /docs)
cd backend
.venv\Scripts\python -m uvicorn app.main:app --port 8000

# terminal 2 - UI on http://localhost:5173  (proxies /api -> :8000)
cd frontend
npm run dev
```

The UI's status pill turns green once the model has loaded.

## Tests

```powershell
cd backend
.venv\Scripts\python -m pytest
```

The tests exercise the real model end to end (probe extraction, measurement,
steering, persistence), so the first run also downloads GPT-2 small.

## How it works

**Probe extraction** (`POST /extract_probe`). For each prompt, run the model and take
the residual stream after block *L* (`blocks.L.hook_resid_post`), mean-pooled over
token positions (the BOS token is excluded; its residual is a huge-norm attention
sink that would otherwise dominate). Average over the positive prompts, average
over the negative prompts, subtract, normalise. Stored with the raw difference
norm plus how well the direction separates its own training prompts.

**Measurement** (`POST /measure`). One forward pass on the new prompt; the pooled
activation at each layer and the per-token activations at the probe layer are
projected onto every requested probe. The UI shows the score on a scale where
0 is the training negatives' mean and 1 the positives' mean.

**Steering** (`POST /generate_with_steering`). Generates twice from the same
prompt and seed: once untouched, once with a forward hook that adds
`coefficient x ||pos - neg|| x direction` to the residual stream at the probe
layer on every token. Coefficient +1 therefore adds exactly one
"positive minus negative" shift; negative values suppress. Both outputs are then
re-measured against the probe at every layer. Each run is logged.

| Endpoint | Purpose |
|---|---|
| `GET /health` | model load status, layer count, device |
| `POST /extract_probe` | build and save a probe from contrastive prompts |
| `GET /probes`, `GET /probes/{id}`, `DELETE /probes/{id}` | saved probes (list omits the vector) |
| `GET /prompt_sets`, `GET /prompt_sets/{id}` | the prompt sets behind each probe |
| `POST /measure` | project a prompt onto saved probes |
| `POST /generate_with_steering` | baseline vs. intervened generation |
| `GET /runs`, `GET /runs/{id}` | intervention run log |

Environment variables: `MI_MODEL` (default `gpt2-small`, any TransformerLens
model name), `MI_DB_PATH` (default `backend/data/sandbox.db`).

## A note on faithfulness

This toy will *feel* like it works even when the underlying phenomenon is shakier
than it looks. A GPT-2-small demo proves the mechanism, not that the interpretation
is "true". The steering panel says so on screen: the intervened output is not
necessarily *right*, just *different*. Perfect separation on the training prompts is
expected (the direction was built from them); only measurement on new prompts says
anything about generalisation.

## Not built yet (stretch)

SAE explorer, Jacobian saliency directions, and the activation verbalizer from the
architecture sketch. A Jacobian direction is just another probe vector, so it would
plug into the existing measure / steer endpoints once extracted.
