import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import ProbeBuilder from "./components/ProbeBuilder";
import ProbeList from "./components/ProbeList";
import MeasurePanel from "./components/MeasurePanel";
import SteerPanel from "./components/SteerPanel";

const TABS = [
  { key: "build", label: "1 · Build probe" },
  { key: "measure", label: "2 · Measure" },
  { key: "steer", label: "3 · Steer" },
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [probes, setProbes] = useState([]);
  const [tab, setTab] = useState("build");
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);

  const refreshProbes = useCallback(async () => {
    try {
      setProbes(await api.listProbes());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  // Poll /health until the model is loaded (or the backend reports a load error).
  useEffect(() => {
    let cancelled = false;
    let timer;
    const poll = async () => {
      try {
        const h = await api.health();
        if (cancelled) return;
        setHealth(h);
        if (!h.model_loaded && h.status !== "error") timer = setTimeout(poll, 2000);
        else refreshProbes();
      } catch (err) {
        if (cancelled) return;
        setHealth({ status: "error", model_loaded: false, error: `backend unreachable: ${err.message}` });
        timer = setTimeout(poll, 3000);
      }
    };
    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [refreshProbes]);

  const modelReady = Boolean(health?.model_loaded);

  const onCreated = (probe) => {
    setSelected(probe.id);
    refreshProbes();
  };

  const onDelete = async (id) => {
    try {
      await api.deleteProbe(id);
      if (selected === id) setSelected(null);
      refreshProbes();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Mech Interp Sandbox</h1>
          <span className="sub">linear probes · emotion vectors · activation steering on a toy model</span>
        </div>
        <StatusPill health={health} />
      </header>

      <div className="layout">
        <aside className="sidebar stack">
          <div>
            <h3>Saved probes</h3>
            <p className="hint">{probes.length} saved · click to select for steering</p>
          </div>
          <ProbeList probes={probes} selectedId={selected} onSelect={(id) => setSelected(id)} onDelete={onDelete} />
        </aside>

        <main className="main">
          <nav className="tabs">
            {TABS.map((t) => (
              <button key={t.key} className={`tab${tab === t.key ? " active" : ""}`} onClick={() => setTab(t.key)}>
                {t.label}
              </button>
            ))}
          </nav>
          {error && <div className="error banner">{error}</div>}
          {tab === "build" && <ProbeBuilder nLayers={health?.n_layers} modelReady={modelReady} onCreated={onCreated} />}
          {tab === "measure" && <MeasurePanel probes={probes} modelReady={modelReady} />}
          {tab === "steer" && (
            <SteerPanel probes={probes} selectedProbeId={selected} onSelectProbe={setSelected} modelReady={modelReady} />
          )}
        </main>
      </div>
    </div>
  );
}

function StatusPill({ health }) {
  if (!health) {
    return (
      <span className="pill">
        <span className="dot warning" /> connecting…
      </span>
    );
  }
  if (health.status === "ok") {
    return (
      <span className="pill">
        <span className="dot good" /> {health.model_name} · {health.n_layers} layers · d={health.d_model} · {health.device}
      </span>
    );
  }
  if (health.status === "loading") {
    return (
      <span className="pill">
        <span className="dot warning" /> loading {health.model_name}…
      </span>
    );
  }
  return (
    <span className="pill" title={health.error ?? ""}>
      <span className="dot critical" /> error: {health.error ?? "unknown"}
    </span>
  );
}
