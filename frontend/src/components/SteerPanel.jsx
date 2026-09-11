import { useEffect, useState } from "react";
import { api } from "../api";
import { LayerColumns, LayerTable, Legend, fmt, signed } from "./charts";

const SERIES_COLORS = { baseline: "var(--s1)", intervened: "var(--s2)" };

export default function SteerPanel({ probes, selectedProbeId, onSelectProbe, modelReady }) {
  const [prompt, setPrompt] = useState("I opened the door and saw");
  const [coefficient, setCoefficient] = useState(2);
  const [maxNewTokens, setMaxNewTokens] = useState(40);
  const [temperature, setTemperature] = useState(0.7);
  const [seed, setSeed] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [run, setRun] = useState(null);
  const [runs, setRuns] = useState([]);

  const probe = probes.find((p) => p.id === selectedProbeId) ?? probes[0] ?? null;

  useEffect(() => {
    if (!selectedProbeId && probes[0]) onSelectProbe(probes[0].id);
  }, [selectedProbeId, probes, onSelectProbe]);

  const refreshRuns = async () => {
    try {
      setRuns(await api.listRuns(20));
    } catch {
      /* history is best-effort */
    }
  };
  useEffect(() => {
    refreshRuns();
  }, []);

  const generate = async () => {
    if (!probe || busy || !prompt.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const res = await api.steer({
        probe_id: probe.id,
        prompt,
        coefficient,
        max_new_tokens: maxNewTokens,
        temperature,
        seed,
      });
      setRun(res);
      refreshRuns();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const onKey = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") generate();
  };

  const norm = probe?.metadata?.norm;
  const runProbe = run ? probes.find((p) => p.id === run.probe_id) : null;
  const traceSeries = run
    ? [
        { name: "baseline", values: run.activation_trace.map((t) => t.baseline), color: SERIES_COLORS.baseline },
        { name: "intervened", values: run.activation_trace.map((t) => t.intervened), color: SERIES_COLORS.intervened },
      ]
    : null;

  return (
    <div className="stack">
      <div className="caveat">
        <b>Read the results honestly.</b> The intervened output is not necessarily <i>right</i>, just{" "}
        <i>different</i>. A GPT-2-small demo proves the mechanism works; it does not prove the
        direction means what its name says. Large coefficients usually produce degenerate text.
      </div>

      <div className="card stack">
        <div>
          <h2>Steer generation</h2>
          <p className="hint">
            Adds coefficient × ‖pos − neg‖ × direction to the residual stream at the probe's layer, on every
            token, during generation. Negative suppresses, positive boosts. Baseline and intervened runs share
            the same seed.
          </p>
        </div>

        <div className="row">
          <label className="field grow">
            Probe
            <select value={probe?.id ?? ""} onChange={(e) => onSelectProbe(e.target.value)} disabled={!probes.length}>
              {!probes.length && <option value="">no probes saved</option>}
              {probes.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} · layer {p.layer}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            Max new tokens
            <input type="number" min="1" max="200" value={maxNewTokens} onChange={(e) => setMaxNewTokens(Number(e.target.value))} />
          </label>
          <label className="field nowrap">
            Temperature · 0 = greedy
            <input type="number" min="0" max="2" step="0.1" value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
          </label>
          <label className="field">
            Seed
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </label>
        </div>

        <label className="field">
          Coefficient · {signed(coefficient)} {norm ? `→ adds ${signed(coefficient * norm, 1)} along the direction` : ""}
          <div className="row">
            <input
              type="range"
              min="-5"
              max="5"
              step="0.25"
              value={coefficient}
              onChange={(e) => setCoefficient(Number(e.target.value))}
              className="grow"
            />
            <input type="number" step="0.25" min="-20" max="20" value={coefficient} onChange={(e) => setCoefficient(Number(e.target.value))} style={{ width: 80 }} />
          </div>
          <div className="row range-labels">
            <span>suppress</span>
            <span>boost</span>
          </div>
        </label>

        <label className="field">
          Prompt
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={onKey} rows={3} />
        </label>

        <div className="row">
          <button className="primary" onClick={generate} disabled={busy || !modelReady || !probe}>
            {busy ? "Generating…" : "Generate baseline vs. intervened"}
          </button>
          <span className="hint">Ctrl+Enter · two generations on CPU take a few seconds</span>
          {error && <span className="error">{error}</span>}
        </div>
      </div>

      {run && (
        <div className={`stack${busy ? " stale" : ""}`}>
          <div className="grid-2">
            <OutputCard
              title="Baseline"
              subtitle="no intervention"
              color={SERIES_COLORS.baseline}
              prompt={run.prompt}
              completion={run.params.completion_baseline ?? run.output_baseline}
            />
            <OutputCard
              title="Intervened"
              subtitle={`${run.mode} · coefficient ${signed(run.coefficient)} at layer ${run.params.layer}`}
              color={SERIES_COLORS.intervened}
              prompt={run.prompt}
              completion={run.params.completion_intervened ?? run.output_intervened}
            />
          </div>

          <div className="card stack">
            <div>
              <h3>Does the output itself read as more “{runProbe?.name ?? "the concept"}”?</h3>
              <p className="hint">
                Both full texts (prompt + completion) are re-run without any hook and projected onto the
                probe at every layer. A gap at the probe layer means the steered text scores differently
                on the very direction that was added, which is a weak sanity check, not proof of meaning.
              </p>
            </div>
            <Legend series={traceSeries} />
            <LayerColumns series={traceSeries} highlightLayer={run.params.layer} />
            <details>
              <summary className="hint">table view</summary>
              <LayerTable series={traceSeries} />
            </details>
          </div>
        </div>
      )}

      {runs.length > 0 && (
        <div className="card stack">
          <h3>Recent runs</h3>
          <ul className="run-list">
            {runs.map((r) => {
              const p = probes.find((x) => x.id === r.probe_id);
              return (
                <li key={r.id} className={`run-row${run?.id === r.id ? " selected" : ""}`} onClick={() => setRun(r)}>
                  <span className="run-meta">
                    {p?.name ?? "deleted probe"} · {r.mode} {signed(r.coefficient)} · T={r.params.temperature}
                  </span>
                  <span className="run-prompt">{r.prompt}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

function OutputCard({ title, subtitle, color, prompt, completion }) {
  return (
    <div className="card stack-sm">
      <div className="row">
        <span className="swatch dot" style={{ background: color }} />
        <b>{title}</b>
        <span className="hint">{subtitle}</span>
      </div>
      <div className="output">
        <span className="prompt">{prompt}</span>
        {completion}
      </div>
      <div className="hint">{completion.trim() ? `${completion.length} chars` : "empty completion (hit EOS immediately)"}</div>
    </div>
  );
}

export { fmt };
