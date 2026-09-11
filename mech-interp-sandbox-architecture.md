# Mech Interp Sandbox — Architecture Sketch

A tool for building and testing the techniques from the ACX piece (linear probes, SAEs, activation verbalizers, emotion/persona vectors, Jacobian saliency) against a small open-weight model, with a UI for building probes and running interventions live.

## Scope

**MVP (weekend-to-week project):** linear probes + emotion/persona vectors + basic steering. These share the same underlying mechanism (contrastive activation averaging), so building one buys you both.

**Stretch:** SAE training/exploration, Jacobian saliency maps, activation verbalizer.

**Out of scope for a personal project:** replicating anything at Claude/GPT scale. Everything below targets a small open model — GPT-2 small (124M) or Pythia-160m — which is enough to demonstrate every technique and runs on a laptop CPU or a free-tier GPU.

## Stack

- **Model + hooks:** `TransformerLens` (purpose-built for exactly this — clean API for grabbing and patching activations at any layer/head). Alternative: `nnsight`, which is more general but has a steeper learning curve.
- **Backend:** Python, FastAPI. Model stays loaded in memory in the backend process; frontend never touches weights directly.
- **Frontend:** React. Doesn't need to be fancy — this is a lab tool, not a product.
- **Storage:** SQLite is plenty. You're storing prompt sets, probe vectors (just arrays of floats), and run logs.

## Core data model

```
PromptSet
  - id, name
  - positive_prompts: [str]   # "the cat sat on the mat", "she pet her cat"...
  - negative_prompts: [str]   # contrastive/control prompts

Probe
  - id, name, prompt_set_id
  - layer: int
  - vector: float[]           # the extracted direction
  - metadata: {norm, extraction_method}

InterventionRun
  - id, probe_id, prompt
  - mode: "measure" | "suppress" | "boost"
  - coefficient: float
  - output_baseline: str
  - output_intervened: str
  - activation_trace: [{layer, projection_value}]
```

## Component 1: Probe builder (covers linear probes + emotion vectors)

Both techniques are the same operation: average activations over "concept present" prompts, average over "concept absent" prompts, subtract.

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small")

def extract_probe(positive_prompts, negative_prompts, layer=6):
    def get_acts(prompts):
        acts = []
        for p in prompts:
            _, cache = model.run_with_cache(p)
            # mean-pool over token positions at the target layer
            acts.append(cache["resid_post", layer].mean(dim=1))
        return torch.stack(acts).mean(dim=0)

    pos_vec = get_acts(positive_prompts)
    neg_vec = get_acts(negative_prompts)
    direction = pos_vec - neg_vec
    return direction / direction.norm()
```

This single function gives you both "cat direction" probes and "desperation" / "calm" emotion vectors — the only difference is what prompt sets you feed it. The UI here is basically: two textareas (positive/negative examples), a layer slider, an "extract" button, and a resulting vector you can name and save.

## Component 2: Live measurement (does the model "think about X" right now?)

Given a saved probe, project a new prompt's activations onto it and show the score:

```python
def measure(probe_vector, prompt, layer=6):
    _, cache = model.run_with_cache(prompt)
    act = cache["resid_post", layer].mean(dim=1)
    return torch.dot(act.squeeze(), probe_vector).item()
```

UI: type a prompt, see a bar per saved probe showing how strongly it activates. This alone reproduces the "does Claude know it's being evaluated" experiment from the article, just at toy scale.

## Component 3: Intervention / steering panel

This is the part that makes the tool feel alive — suppress or boost a direction during generation and watch behavior change, same as the blackmail-experiment or eval-awareness-suppression examples in the article.

```python
def make_steering_hook(direction, coefficient):
    def hook(activation, hook):
        return activation + coefficient * direction
    return hook

def generate_with_steering(prompt, probe_vector, layer, coefficient):
    hook_fn = make_steering_hook(probe_vector, coefficient)
    with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", hook_fn)]):
        output = model.generate(prompt, max_new_tokens=50)
    return output
```

UI: pick a probe, a coefficient slider (negative = suppress, positive = boost), a prompt box, and side-by-side baseline vs. intervened output.

## Component 4 (stretch): SAE explorer

Train a small sparse autoencoder on a layer's activations (there are open implementations to adapt — SAELens is the standard one). The app-side work isn't training it (that's an offline batch job) — it's the *browsing* UI: given a trained SAE, show which of its features fire on a given prompt, and let the user click a feature to see the top prompts that activate it, then let them ablate that feature and regenerate. This is genuinely the most compute-heavy piece and the one I'd defer.

## Component 5 (stretch): Jacobian saliency

```python
def jacobian_direction(target_token_id, layer, prompts):
    grads = []
    for p in prompts:
        tokens = model.to_tokens(p)
        tokens.requires_grad_(False)
        logits, cache = model.run_with_cache(tokens)
        act = cache["resid_post", layer]
        act.retain_grad()
        logits[0, -1, target_token_id].backward(retain_graph=True)
        grads.append(act.grad.mean(dim=1))
    return torch.stack(grads).mean(dim=0)
```

Same measurement/intervention UI as Component 2 can consume this — a Jacobian direction is just another kind of probe vector once extracted.

## Component 6 (stretch): Activation verbalizer

This is the one real training project in the list: fine-tune a small model (even the same GPT-2) to take an activation vector as a soft-prompt-style input and output a natural-language description. Realistically this needs a labeled dataset of (activation, description) pairs, which you'd bootstrap by having a larger model (e.g. via API) generate descriptions of prompts, then pairing those with the small model's activations on the same prompts. Worth scoping as its own mini-project once the rest of the tool works.

## Suggested build order

1. Backend: load GPT-2 small in TransformerLens, expose `/extract_probe`, `/measure`, `/generate_with_steering` as FastAPI endpoints.
2. Frontend: probe builder form → save/list probes → measurement view.
3. Add the steering panel with baseline-vs-intervened side-by-side.
4. Ship that as v1 — it already demonstrates 3 of the 6 techniques (probes, emotion vectors, and a form of steering) end to end.
5. Only then decide if SAEs, Jacobians, or the verbalizer are worth the extra compute/training investment.

## Notes on faithfulness to the real thing

Worth being upfront with yourself (or anyone you show this to) that this toy version will *feel* like it works even when the underlying phenomenon is shakier than it looks — that's literally the plot of the article: SAEs and probes looked crisp on tiny models and got muddled at scale. A GPT-2-small demo proves the mechanism, not that the interpretation is "true." That caveat is itself a good thing to bake into the UI (e.g., always show the intervened output isn't necessarily *right*, just *different*).
