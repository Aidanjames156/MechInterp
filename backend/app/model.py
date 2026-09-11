"""Model service for the Mech Interp Sandbox.

Wraps a TransformerLens ``HookedTransformer`` and implements the three core
operations from the architecture sketch:

* ``extract_probe`` - contrastive activation averaging. Linear probes and
  emotion/persona vectors are the same operation fed different prompt sets.
* ``measure``       - project a prompt's activations onto saved directions.
* ``steer``         - add / subtract a direction during generation and return
  baseline vs. intervened output side by side.

All model access is serialised with a lock: hooks are global state on the
model object, so two requests must never run a forward pass concurrently.
"""
from __future__ import annotations

import threading
from typing import Callable, Literal

import torch
from transformer_lens import HookedTransformer

Pooling = Literal["mean", "last"]


def resid_hook_name(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"


def _pool(acts: torch.Tensor, pooling: Pooling) -> torch.Tensor:
    """Pool a ``[pos, d_model]`` activation tensor down to ``[d_model]``.

    Position 0 is the BOS token TransformerLens prepends by default. Its
    residual stream is a huge-norm outlier (it acts as an attention sink) that
    would otherwise dominate a mean, so it is skipped whenever there is
    anything else to pool.
    """
    body = acts[1:] if acts.shape[0] > 1 else acts
    if pooling == "last":
        return body[-1]
    return body.mean(dim=0)


def make_steering_hook(direction: torch.Tensor, scale: float) -> Callable:
    """Forward hook that adds ``scale * direction`` to the residual stream.

    The BOS position is left untouched on the full-prompt pass. During
    KV-cached generation each later pass carries a single new token, which
    is always steered.
    """

    def hook(resid: torch.Tensor, hook):  # resid: [batch, pos, d_model]
        if resid.shape[1] > 1:
            resid = resid.clone()
            resid[:, 1:, :] = resid[:, 1:, :] + scale * direction
            return resid
        return resid + scale * direction

    return hook


class ModelService:
    def __init__(self, model_name: str = "gpt2-small", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model: HookedTransformer | None = None
        self.load_error: str | None = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ setup
    def load(self) -> HookedTransformer:
        with self._lock:
            if self.model is None:
                try:
                    model = HookedTransformer.from_pretrained(self.model_name, device=self.device)
                except Exception as exc:  # surfaced via /health
                    self.load_error = f"{type(exc).__name__}: {exc}"
                    raise
                model.eval()
                self.model = model
                self.load_error = None
        return self.model

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _require(self) -> HookedTransformer:
        if self.model is None:
            raise RuntimeError("model is not loaded")
        return self.model

    @property
    def n_layers(self) -> int:
        return self._require().cfg.n_layers

    @property
    def d_model(self) -> int:
        return self._require().cfg.d_model

    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "n_layers": self.n_layers if self.loaded else None,
            "d_model": self.d_model if self.loaded else None,
        }

    def check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise ValueError(f"layer must be in [0, {self.n_layers - 1}], got {layer}")

    # ------------------------------------------------------------ activations
    def _resid_stack(self, prompt: str) -> tuple[list[str], torch.Tensor]:
        """One forward pass -> (str_tokens, resid_post stack ``[n_layers, pos, d_model]``)."""
        model = self._require()
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_resid_post"))
        stack = torch.stack([cache[resid_hook_name(layer)][0] for layer in range(model.cfg.n_layers)])
        return str_tokens, stack

    def _pooled_at_layer(self, prompt: str, layer: int, pooling: Pooling) -> torch.Tensor:
        model = self._require()
        name = resid_hook_name(layer)
        _, cache = model.run_with_cache(model.to_tokens(prompt), names_filter=lambda n: n == name)
        return _pool(cache[name][0], pooling)

    # ------------------------------------------------------- component 1: probe
    def extract_probe(
        self,
        positive_prompts: list[str],
        negative_prompts: list[str],
        layer: int,
        pooling: Pooling = "mean",
    ) -> tuple[list[float], dict]:
        """Mean activation over positives minus mean over negatives, unit-normalised.

        Returns ``(direction, metadata)``. ``metadata["norm"]`` is the norm of
        the raw difference, which the steering hook uses as its unit of scale:
        coefficient 1.0 adds exactly one "positive minus negative" shift.
        """
        self.check_layer(layer)
        with self._lock, torch.inference_mode():
            pos = torch.stack([self._pooled_at_layer(p, layer, pooling) for p in positive_prompts])
            neg = torch.stack([self._pooled_at_layer(p, layer, pooling) for p in negative_prompts])
            diff = pos.mean(dim=0) - neg.mean(dim=0)
            raw_norm = diff.norm().item()
            if raw_norm < 1e-8:
                raise ValueError(
                    "positive and negative prompts have identical mean activations; no direction to extract"
                )
            direction = diff / raw_norm

            pos_proj = pos @ direction
            neg_proj = neg @ direction
            pos_mean = pos_proj.mean().item()
            neg_mean = neg_proj.mean().item()
            threshold = (pos_mean + neg_mean) / 2
            correct = (pos_proj > threshold).sum().item() + (neg_proj <= threshold).sum().item()
            mean_resid_norm = torch.cat([pos, neg]).norm(dim=-1).mean().item()

        metadata = {
            "norm": raw_norm,
            "extraction_method": "mean_difference",
            "pooling": pooling,
            "model": self.model_name,
            "n_positive": len(positive_prompts),
            "n_negative": len(negative_prompts),
            "pos_mean_projection": pos_mean,
            "neg_mean_projection": neg_mean,
            "threshold": threshold,
            "train_accuracy": correct / (len(positive_prompts) + len(negative_prompts)),
            "mean_resid_norm": mean_resid_norm,
            "pos_projections": pos_proj.tolist(),
            "neg_projections": neg_proj.tolist(),
        }
        return direction.cpu().tolist(), metadata

    # ----------------------------------------------------- component 2: measure
    def measure(self, probes: list[dict], prompt: str, pooling: Pooling = "mean") -> dict:
        """Project one prompt onto many probes with a single forward pass.

        ``probes`` is a list of ``{"vector": [...], "layer": int}``. For each
        probe the result carries the pooled projection at the probe's own
        layer (``score``), the cosine between pooled activation and direction,
        the pooled projection at every layer, and the per-token projection at
        the probe's layer (index 0 is the BOS token).
        """
        for pr in probes:
            self.check_layer(pr["layer"])
        with self._lock, torch.inference_mode():
            str_tokens, stack = self._resid_stack(prompt)
            pooled = torch.stack([_pool(stack[layer], pooling) for layer in range(stack.shape[0])])
            results = []
            for pr in probes:
                layer = pr["layer"]
                v = torch.tensor(pr["vector"], device=self.device, dtype=stack.dtype)
                per_layer = (pooled @ v).tolist()
                per_token = (stack[layer] @ v).tolist()
                cosine = torch.nn.functional.cosine_similarity(pooled[layer], v, dim=0).item()
                results.append({
                    "score": per_layer[layer],
                    "cosine": cosine,
                    "per_layer": per_layer,
                    "per_token": per_token,
                })
        return {"tokens": str_tokens, "results": results}

    # ------------------------------------------------------ component 3: steer
    def _generate(
        self, prompt: str, max_new_tokens: int, temperature: float, seed: int, fwd_hooks: list
    ) -> str:
        model = self._require()
        do_sample = temperature > 0
        torch.manual_seed(seed)
        with model.hooks(fwd_hooks=fwd_hooks):
            return model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                stop_at_eos=True,
                verbose=False,
                return_type="str",
            )

    @staticmethod
    def _completion(prompt: str, full_text: str) -> str:
        return full_text[len(prompt):] if full_text.startswith(prompt) else full_text

    def steer(
        self,
        vector: list[float],
        layer: int,
        raw_norm: float,
        prompt: str,
        coefficient: float,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        seed: int = 0,
        pooling: Pooling = "mean",
    ) -> dict:
        """Generate twice from the same prompt and seed: once untouched, once with
        ``coefficient * raw_norm * direction`` added to ``resid_post`` at ``layer``.

        Both outputs are then measured (without hooks) against the same probe at
        every layer, so the caller can see whether the produced text itself reads
        as more or less of the concept.
        """
        self.check_layer(layer)
        with self._lock, torch.inference_mode():
            direction = torch.tensor(vector, device=self.device)
            scale = coefficient * raw_norm
            hook = make_steering_hook(direction, scale)
            baseline = self._generate(prompt, max_new_tokens, temperature, seed, fwd_hooks=[])
            intervened = self._generate(
                prompt, max_new_tokens, temperature, seed, fwd_hooks=[(resid_hook_name(layer), hook)]
            )
        probe = [{"vector": vector, "layer": layer}]
        trace_b = self.measure(probe, baseline, pooling)["results"][0]["per_layer"]
        trace_i = self.measure(probe, intervened, pooling)["results"][0]["per_layer"]
        return {
            "output_baseline": baseline,
            "output_intervened": intervened,
            "completion_baseline": self._completion(prompt, baseline),
            "completion_intervened": self._completion(prompt, intervened),
            "effective_scale": scale,
            "activation_trace": [
                {"layer": i, "baseline": b, "intervened": v}
                for i, (b, v) in enumerate(zip(trace_b, trace_i))
            ],
        }
