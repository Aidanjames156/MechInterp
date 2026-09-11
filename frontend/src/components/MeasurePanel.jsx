import { useState } from "react";
import { api } from "../api";
import { LayerColumns, LayerTable, ScoreBar, TokenStrip, fmt } from "./charts";

export default function MeasurePanel({ probes, modelReady }) {
  const [prompt, setPrompt] = useState("The kitten curled up by the fire and purred.");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [open, setOpen] = useState(null);

  const run = async () => {
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.measure({ prompt });
      setResult(res);
      if (res.results.length && !res.results.some((r) => r.probe_id === open)) setOpen(res.results[0].probe_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const onKey = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run();
  };

  return (
    <div className="stack">
      <div className="card stack">
        <div>
          <h2>Live measurement</h2>
          <p className="hint">
            Project a new prompt onto every saved probe. On each bar, 0 is where the probe's
            training negatives landed on average and 1 where the positives did; anything past
            those marks is more extreme than the training examples.
          </p>
        </div>
        <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={onKey} rows={3} />
        <div className="row">
          <button className="primary" onClick={run} disabled={busy || !modelReady || probes.length === 0}>
            {busy ? "Measuring…" : "Measure"}
          </button>
          <span className="hint">Ctrl+Enter</span>
          {probes.length === 0 && <span className="hint">build a probe first</span>}
          {error && <span className="error">{error}</span>}
        </div>
      </div>

      {result && result.results.length > 0 && (
        <div className={`card stack${busy ? " stale" : ""}`}>
          <h3>Probe activations · {result.tokens.length - 1} tokens</h3>
          {result.results.map((r) => {
            const isOpen = open === r.probe_id;
            const threshold = (r.pos_mean_projection + r.neg_mean_projection) / 2;
            return (
              <div key={r.probe_id} className={`measure${isOpen ? " open" : ""}`}>
                <div className="measure-row" onClick={() => setOpen(isOpen ? null : r.probe_id)}>
                  <div className="measure-name">
                    <b>{r.probe_name}</b>
                    <span className="hint"> L{r.layer}</span>
                  </div>
                  <ScoreBar value={r.normalized} />
                  <div className="measure-value">{fmt(r.normalized)}</div>
                </div>
                {isOpen && (
                  <div className="measure-detail stack">
                    <div className="hint">
                      raw projection {fmt(r.score, 1)} · cosine {fmt(r.cosine, 3)} · training means: negatives{" "}
                      {fmt(r.neg_mean_projection, 1)}, positives {fmt(r.pos_mean_projection, 1)}
                    </div>
                    <div>
                      <h3>Per token · layer {r.layer}</h3>
                      <TokenStrip tokens={result.tokens} values={r.per_token} center={threshold} />
                    </div>
                    <div>
                      <h3>Pooled projection by layer</h3>
                      <LayerColumns
                        series={[{ name: "projection", values: r.per_layer, color: "var(--s1)" }]}
                        highlightLayer={r.layer}
                        height={150}
                      />
                      <details>
                        <summary className="hint">table view</summary>
                        <LayerTable series={[{ name: "projection", values: r.per_layer }]} />
                      </details>
                      <p className="hint">
                        The probe was extracted at layer {r.layer}; projections at other layers are shown
                        for context only, since the residual stream's scale and basis drift across layers.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
