import { useState } from "react";
import { api } from "../api";
import { PRESETS } from "../presets";
import { Legend, SeparationDots, fmt } from "./charts";

const toLines = (text) =>
  text
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);

const SERIES = [
  { name: "positive prompts", color: "var(--s1)" },
  { name: "negative prompts", color: "var(--s2)" },
];

export default function ProbeBuilder({ nLayers, modelReady, onCreated }) {
  const maxLayer = Math.max(0, (nLayers ?? 12) - 1);
  const [name, setName] = useState("");
  const [pos, setPos] = useState("");
  const [neg, setNeg] = useState("");
  const [layer, setLayer] = useState(Math.min(6, maxLayer));
  const [pooling, setPooling] = useState("mean");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const nPos = toLines(pos).length;
  const nNeg = toLines(neg).length;

  const applyPreset = (key) => {
    const p = PRESETS.find((x) => x.key === key);
    if (!p) return;
    setName(p.name);
    setPos(p.positive.join("\n"));
    setNeg(p.negative.join("\n"));
  };

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const positive_prompts = toLines(pos);
      const negative_prompts = toLines(neg);
      const probe = await api.extractProbe({
        name: name.trim() || "untitled probe",
        positive_prompts,
        negative_prompts,
        layer,
        pooling,
      });
      setResult({ probe, positive_prompts, negative_prompts });
      onCreated(probe);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const meta = result?.probe.metadata;

  return (
    <div className="stack">
      <form className="card stack" onSubmit={submit}>
        <div>
          <h2>Build a probe</h2>
          <p className="hint">
            Mean activation over the concept-present prompts, minus the mean over the
            concept-absent prompts, at one layer of the residual stream. Feed it a topic
            contrast and you get a linear probe; feed it an emotional or stylistic contrast
            and the same operation gives you an emotion / persona vector.
          </p>
        </div>

        <div className="row">
          <label className="field grow">
            Probe name
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. cats" />
          </label>
          <label className="field">
            Load a preset
            <select defaultValue="" onChange={(e) => applyPreset(e.target.value)}>
              <option value="" disabled>
                choose…
              </option>
              {PRESETS.map((p) => (
                <option key={p.key} value={p.key}>
                  {p.name} · {p.kind}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="grid-2">
          <label className="field">
            Positive prompts · concept present <span className="count">{nPos}</span>
            <textarea value={pos} onChange={(e) => setPos(e.target.value)} placeholder={"one prompt per line"} />
          </label>
          <label className="field">
            Negative prompts · concept absent / control <span className="count">{nNeg}</span>
            <textarea value={neg} onChange={(e) => setNeg(e.target.value)} placeholder={"one prompt per line"} />
          </label>
        </div>

        <div className="row">
          <label className="field grow">
            Layer · resid_post at block {layer} of {maxLayer + 1}
            <input type="range" min="0" max={maxLayer} value={layer} onChange={(e) => setLayer(Number(e.target.value))} />
          </label>
          <label className="field">
            Pooling over tokens
            <select value={pooling} onChange={(e) => setPooling(e.target.value)}>
              <option value="mean">mean (excl. BOS)</option>
              <option value="last">last token</option>
            </select>
          </label>
        </div>

        <div className="row">
          <button className="primary" type="submit" disabled={busy || !modelReady || nPos === 0 || nNeg === 0}>
            {busy ? "Extracting…" : "Extract & save probe"}
          </button>
          {!modelReady && <span className="hint">waiting for the model to load</span>}
          {error && <span className="error">{error}</span>}
        </div>
      </form>

      {result && (
        <div className="card stack">
          <div>
            <h2>
              Saved “{result.probe.name}” · layer {result.probe.layer} · {meta.pooling} pooling
            </h2>
            <p className="hint">
              Unit direction in a {result.probe.vector.length}-dimensional residual stream. Steering
              coefficient 1.0 adds exactly this positive-minus-negative shift (norm {fmt(meta.norm, 1)}).
            </p>
          </div>
          <div className="stats">
            <Stat label="Train accuracy" value={`${Math.round(meta.train_accuracy * 100)}%`} hint="midpoint threshold on the training prompts" />
            <Stat label="Separation" value={fmt(meta.pos_mean_projection - meta.neg_mean_projection, 1)} hint="pos mean − neg mean projection" />
            <Stat label="‖pos − neg‖" value={fmt(meta.norm, 1)} hint="raw difference norm" />
            <Stat label="Mean resid norm" value={fmt(meta.mean_resid_norm, 1)} hint="typical activation size at this layer" />
          </div>
          <div>
            <h3>Training prompt projections</h3>
            <Legend series={SERIES} />
            <SeparationDots
              positives={meta.pos_projections}
              negatives={meta.neg_projections}
              threshold={meta.threshold}
              positivePrompts={result.positive_prompts}
              negativePrompts={result.negative_prompts}
            />
            <p className="hint">
              Perfect separation on the training prompts is expected and proves little: the
              direction was built from them. Measure on new prompts to see if it generalises.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="hint">{hint}</div>}
    </div>
  );
}
