export default function ProbeList({ probes, selectedId, onSelect, onDelete }) {
  if (!probes.length) {
    return <p className="hint">No probes yet. Build one from the first tab.</p>;
  }
  return (
    <ul className="probe-list">
      {probes.map((p) => {
        const acc = p.metadata?.train_accuracy;
        return (
          <li
            key={p.id}
            className={`probe-row${p.id === selectedId ? " selected" : ""}`}
            onClick={() => onSelect(p.id)}
          >
            <div>
              <div className="name">{p.name}</div>
              <div className="meta">
                layer {p.layer} · {p.metadata?.pooling ?? "mean"} · {p.metadata?.n_positive ?? "?"}+
                {p.metadata?.n_negative ?? "?"} prompts
                {acc !== undefined && ` · ${Math.round(acc * 100)}% train`}
              </div>
            </div>
            <button
              className="ghost small"
              title="Delete probe and its runs"
              onClick={(e) => {
                e.stopPropagation();
                if (window.confirm(`Delete probe “${p.name}” and its steering runs?`)) onDelete(p.id);
              }}
            >
              ×
            </button>
          </li>
        );
      })}
    </ul>
  );
}
