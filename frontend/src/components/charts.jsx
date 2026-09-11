// Small chart primitives shared by the panels. All colors come from CSS
// custom properties defined in styles.css so light/dark swap in one place.
import { createContext, useCallback, useContext, useMemo, useState } from "react";

// ------------------------------------------------------------------ tooltip
const TooltipCtx = createContext(null);

export function TooltipProvider({ children }) {
  const [tip, setTip] = useState(null);
  const show = useCallback((e, content) => setTip({ x: e.clientX, y: e.clientY, content }), []);
  const move = useCallback((e) => {
    const { clientX, clientY } = e;
    setTip((t) => (t ? { ...t, x: clientX, y: clientY } : t));
  }, []);
  const hide = useCallback(() => setTip(null), []);
  const value = useMemo(() => ({ show, move, hide }), [show, move, hide]);
  return (
    <TooltipCtx.Provider value={value}>
      {children}
      {tip && <TooltipBox {...tip} />}
    </TooltipCtx.Provider>
  );
}

function TooltipBox({ x, y, content }) {
  const left = Math.min(x + 12, window.innerWidth - 340);
  const top = y + 14 > window.innerHeight - 80 ? y - 60 : y + 14;
  return (
    <div className="tooltip" style={{ left, top }}>
      {content}
    </div>
  );
}

const noop = { show: () => {}, move: () => {}, hide: () => {} };
export function useTooltip() {
  return useContext(TooltipCtx) ?? noop;
}

export function TipRows({ title, rows }) {
  return (
    <div>
      {title && <div className="tip-title">{title}</div>}
      {rows.map(([k, v]) => (
        <div key={k} className="tip-row">
          <span>{k}</span>
          <span>{v}</span>
        </div>
      ))}
    </div>
  );
}

// ------------------------------------------------------------------ helpers
export function fmt(v, d = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  const digits = a >= 100 ? 0 : a >= 10 ? 1 : d;
  return v.toFixed(digits);
}

export function signed(v, d = 2) {
  return (v > 0 ? "+" : "") + fmt(v, d);
}

function niceTicks(lo, hi, count = 4) {
  if (hi === lo) hi = lo + 1;
  const rough = (hi - lo) / count;
  const p = Math.pow(10, Math.floor(Math.log10(rough)));
  const candidates = [1, 2, 2.5, 5, 10].map((c) => c * p);
  const step = candidates.find((c) => c >= rough) ?? candidates[candidates.length - 1];
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Number(v.toFixed(10)));
  return ticks;
}

function fmtTick(v) {
  return Math.abs(v) >= 1000 ? v.toLocaleString() : String(v);
}

// A column with a 4px rounded data-end and a square baseline end.
function columnPath(x, yBase, yVal, w, r) {
  const top = Math.min(yBase, yVal);
  const bottom = Math.max(yBase, yVal);
  const h = bottom - top;
  if (h <= 0) return "";
  const rr = Math.min(r, w / 2, h);
  if (yVal <= yBase) {
    return `M${x},${bottom} V${top + rr} Q${x},${top} ${x + rr},${top} H${x + w - rr} Q${x + w},${top} ${x + w},${top + rr} V${bottom} Z`;
  }
  return `M${x},${top} H${x + w} V${bottom - rr} Q${x + w},${bottom} ${x + w - rr},${bottom} H${x + rr} Q${x},${bottom} ${x},${bottom - rr} Z`;
}

// ------------------------------------------------------------------ legend
export function Legend({ series }) {
  if (!series || series.length < 2) return null;
  return (
    <div className="legend">
      {series.map((s) => (
        <span key={s.name} className="legend-item">
          <span className="swatch dot" style={{ background: s.color }} />
          {s.name}
        </span>
      ))}
    </div>
  );
}

// ----------------------------------------------------------------- ScoreBar
// One-series horizontal bar on a fixed [-0.5, 1.5] scale where 0 is the
// training negatives' mean projection and 1 the positives' mean.
const DOMAIN = [-0.5, 1.5];

export function ScoreBar({ value }) {
  const [lo, hi] = DOMAIN;
  const pct = (v) => ((v - lo) / (hi - lo)) * 100;
  const clamped = Math.max(lo, Math.min(hi, value));
  const left = pct(Math.min(0, clamped));
  const width = pct(Math.max(0, clamped)) - pct(Math.min(0, clamped));
  const isClamped = clamped !== value;
  return (
    <div className="scorebar" aria-hidden="true">
      <div className="scorebar-track" />
      <div className="scorebar-threshold" style={{ left: `${pct(0.5)}%` }} />
      <div className="scorebar-tick" style={{ left: `${pct(0)}%` }} />
      <div className="scorebar-tick" style={{ left: `${pct(1)}%` }} />
      <div className="scorebar-tick-label" style={{ left: `${pct(0)}%` }}>neg mean</div>
      <div className="scorebar-tick-label" style={{ left: `${pct(1)}%` }}>pos mean</div>
      <div
        className={`scorebar-fill ${clamped >= 0 ? "pos" : "neg"}${isClamped ? " clamped" : ""}`}
        style={{ left: `${left}%`, width: `${width}%` }}
      />
    </div>
  );
}

// --------------------------------------------------------------- TokenStrip
// Diverging encoding of per-token projection around the probe threshold:
// blue = above (concept present), red = below, gray = at the threshold.
const BOS = "<|endoftext|>";

export function TokenStrip({ tokens, values, center }) {
  const tip = useTooltip();
  const devs = values.map((v) => v - center);
  const maxAbs = Math.max(1e-9, ...devs.map((d, i) => (i === 0 && tokens[i] === BOS ? 0 : Math.abs(d))));
  return (
    <div className="stack-sm">
      <div className="tokens">
        {tokens.map((tok, i) => {
          const isBos = i === 0 && tok === BOS;
          const t = isBos ? 0 : devs[i] / maxAbs;
          const pole = t >= 0 ? "var(--div-pos)" : "var(--div-neg)";
          const strength = Math.round(Math.abs(t) * 60);
          const style = isBos ? undefined : { background: `color-mix(in oklab, ${pole} ${strength}%, var(--div-mid))` };
          const label = isBos ? "⟨bos⟩" : tok.replace(/\n/g, "⏎");
          return (
            <span
              key={i}
              className={`token${isBos ? " bos" : ""}`}
              style={style}
              onPointerEnter={(e) =>
                tip.show(
                  e,
                  <TipRows
                    title={isBos ? "BOS token (attention sink, not scored)" : JSON.stringify(tok)}
                    rows={isBos ? [["projection", fmt(values[i], 1)]] : [
                      ["projection", fmt(values[i], 1)],
                      ["vs threshold", signed(devs[i], 1)],
                    ]}
                  />,
                )
              }
              onPointerMove={tip.move}
              onPointerLeave={tip.hide}
            >
              {label}
            </span>
          );
        })}
      </div>
      <div className="legend scale-legend">
        <span className="legend-item"><span className="swatch" style={{ background: "var(--div-neg)" }} />below threshold</span>
        <span className="legend-item"><span className="swatch" style={{ background: "var(--div-mid)", border: "1px solid var(--axis)" }} />at threshold</span>
        <span className="legend-item"><span className="swatch" style={{ background: "var(--div-pos)" }} />above threshold</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- LayerColumns
// Grouped columns: one group per layer, one column per series.
export function LayerColumns({ series, highlightLayer, height = 180 }) {
  const tip = useTooltip();
  const [hover, setHover] = useState(null);
  const W = 640;
  const H = height;
  const m = { top: 10, right: 8, bottom: 26, left: 48 };
  const n = series[0]?.values.length ?? 0;
  const all = series.flatMap((s) => s.values);
  const lo = Math.min(0, ...all);
  const hi = Math.max(0, ...all);
  const ticks = niceTicks(lo, hi, 4);
  const yLo = Math.min(lo, ticks[0]);
  const yHi = Math.max(hi, ticks[ticks.length - 1]);
  const plotW = W - m.left - m.right;
  const plotH = H - m.top - m.bottom;
  const y = (v) => m.top + plotH - ((v - yLo) / (yHi - yLo || 1)) * plotH;
  const groupW = plotW / Math.max(1, n);
  const k = series.length;
  const gap = 2;
  const colW = Math.max(2, Math.min(24, (groupW - 8 - gap * (k - 1)) / k));
  const groupInner = colW * k + gap * (k - 1);

  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Projection by layer">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={m.left} x2={W - m.right} y1={y(t)} y2={y(t)} stroke={t === 0 ? "var(--axis)" : "var(--grid)"} strokeWidth="1" />
          <text x={m.left - 6} y={y(t)} textAnchor="end" dominantBaseline="middle" fontSize="10" fill="var(--muted)">
            {fmtTick(t)}
          </text>
        </g>
      ))}
      {Array.from({ length: n }, (_, i) => {
        const gx = m.left + i * groupW;
        const x0 = gx + (groupW - groupInner) / 2;
        const isHl = i === highlightLayer;
        return (
          <g key={i}>
            {(isHl || hover === i) && <rect x={gx} y={m.top} width={groupW} height={plotH} fill="var(--wash)" />}
            {series.map((s, j) => (
              <path key={s.name} d={columnPath(x0 + j * (colW + gap), y(0), y(s.values[i]), colW, 4)} fill={s.color} />
            ))}
            <text
              x={gx + groupW / 2}
              y={H - 8}
              textAnchor="middle"
              fontSize="10"
              fill={isHl ? "var(--ink)" : "var(--muted)"}
              fontWeight={isHl ? 600 : 400}
            >
              {i}
            </text>
            <rect
              x={gx}
              y={m.top}
              width={groupW}
              height={plotH + m.bottom}
              fill="transparent"
              pointerEvents="all"
              onPointerEnter={(e) => {
                setHover(i);
                tip.show(
                  e,
                  <TipRows
                    title={`layer ${i}${isHl ? " · probe layer" : ""}`}
                    rows={series.map((s) => [s.name, fmt(s.values[i], 1)])}
                  />,
                );
              }}
              onPointerMove={tip.move}
              onPointerLeave={() => {
                setHover(null);
                tip.hide();
              }}
            />
          </g>
        );
      })}
    </svg>
  );
}

export function LayerTable({ series }) {
  const n = series[0]?.values.length ?? 0;
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>layer</th>
          {series.map((s) => (
            <th key={s.name}>{s.name}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: n }, (_, i) => (
          <tr key={i}>
            <td>{i}</td>
            {series.map((s) => (
              <td key={s.name}>{fmt(s.values[i], 2)}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ----------------------------------------------------------- SeparationDots
// 1-D strip plot of each training prompt's projection onto the extracted
// direction, split by class, with the decision threshold marked.
export function SeparationDots({ positives, negatives, threshold, positivePrompts = [], negativePrompts = [] }) {
  const tip = useTooltip();
  const W = 640;
  const H = 76;
  const m = { left: 70, right: 16 };
  const all = [...positives, ...negatives, threshold];
  let lo = Math.min(...all);
  let hi = Math.max(...all);
  const pad = (hi - lo || 1) * 0.08;
  lo -= pad;
  hi += pad;
  const x = (v) => m.left + ((v - lo) / (hi - lo)) * (W - m.left - m.right);
  const rows = [
    { name: "positive", values: positives, prompts: positivePrompts, color: "var(--s1)", y: 22 },
    { name: "negative", values: negatives, prompts: negativePrompts, color: "var(--s2)", y: 48 },
  ];
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Training prompt projections">
      <line x1={x(threshold)} x2={x(threshold)} y1={8} y2={H - 14} stroke="var(--muted)" strokeWidth="1" />
      <text x={x(threshold)} y={H - 3} fontSize="9" textAnchor="middle" fill="var(--muted)">
        threshold {fmt(threshold, 1)}
      </text>
      {rows.map((r) => (
        <g key={r.name}>
          <text x={m.left - 10} y={r.y} fontSize="11" textAnchor="end" dominantBaseline="middle" fill="var(--ink-2)">
            {r.name}
          </text>
          <line x1={m.left} x2={W - m.right} y1={r.y} y2={r.y} stroke="var(--grid)" strokeWidth="1" />
          {r.values.map((v, i) => (
            <g
              key={i}
              onPointerEnter={(e) =>
                tip.show(e, <TipRows title={r.prompts[i] ?? r.name} rows={[["projection", fmt(v, 1)]]} />)
              }
              onPointerMove={tip.move}
              onPointerLeave={tip.hide}
            >
              <circle cx={x(v)} cy={r.y} r="12" fill="transparent" pointerEvents="all" />
              <circle cx={x(v)} cy={r.y} r="5" fill={r.color} stroke="var(--surface)" strokeWidth="2" />
            </g>
          ))}
        </g>
      ))}
    </svg>
  );
}
