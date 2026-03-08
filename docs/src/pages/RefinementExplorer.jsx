import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import useInView from '../hooks/useInView';

const METHOD_COLORS = {
  production: '#ef4444',
  argmin_y: '#f59e0b',
  argmin_xy: '#d946ef',
  velocity_zero: '#22d3ee',
};

const METHOD_LABELS = {
  production: 'Production (1D argmin)',
  argmin_y: 'argmin(y) — highest point',
  argmin_xy: 'argmin(x+y) — top-left corner',
  velocity_zero: 'Velocity zero-crossing',
};

const CONTACT_COLORS = {
  production: '#ef4444',
  argmax_y: '#f59e0b',
  argmax_xy: '#d946ef',
  argmax_x: '#22d3ee',
};

const CONTACT_LABELS = {
  production: 'Production (1D argmax)',
  argmax_y: 'argmax(y) — lowest point',
  argmax_xy: 'argmax(x+y) — bottom-right',
  argmax_x: 'argmax(x) — furthest forward',
};

const VOTING_LABELS = {
  mean: 'Mean',
  median: 'Median',
  majority: 'Majority Vote',
};

const MARKER_SHAPES = {
  production: 'X',
  argmin_y: 'circle',
  argmin_xy: 'diamond',
  velocity_zero: 'triangle',
  argmax_y: 'circle',
  argmax_xy: 'diamond',
  argmax_x: 'triangle',
};

function MarkerSvg({ shape, x, y, size = 8, color, stroke = 'white', strokeWidth = 2 }) {
  if (shape === 'X') {
    const s = size;
    return (
      <g>
        <line x1={x - s} y1={y - s} x2={x + s} y2={y + s} stroke={color} strokeWidth={strokeWidth + 1} />
        <line x1={x + s} y1={y - s} x2={x - s} y2={y + s} stroke={color} strokeWidth={strokeWidth + 1} />
      </g>
    );
  }
  if (shape === 'circle') return <circle cx={x} cy={y} r={size} fill={color} stroke={stroke} strokeWidth={strokeWidth} />;
  if (shape === 'diamond') {
    const s = size * 1.2;
    return <polygon points={`${x},${y - s} ${x + s},${y} ${x},${y + s} ${x - s},${y}`} fill={color} stroke={stroke} strokeWidth={strokeWidth} />;
  }
  if (shape === 'triangle') {
    const s = size * 1.2;
    return <polygon points={`${x},${y - s} ${x + s},${y + s} ${x - s},${y + s}`} fill={color} stroke={stroke} strokeWidth={strokeWidth} />;
  }
  if (shape === 'star') {
    const outer = size * 1.4, inner = size * 0.6;
    let pts = '';
    for (let i = 0; i < 5; i++) {
      const ao = (i * 72 - 90) * Math.PI / 180;
      const ai = ((i * 72) + 36 - 90) * Math.PI / 180;
      pts += `${x + outer * Math.cos(ao)},${y + outer * Math.sin(ao)} `;
      pts += `${x + inner * Math.cos(ai)},${y + inner * Math.sin(ai)} `;
    }
    return <polygon points={pts} fill={color} stroke={stroke} strokeWidth={strokeWidth} />;
  }
  return null;
}

function computeEnsemble(picks, voting, enabledMethods) {
  const valid = enabledMethods
    .map(m => picks[m])
    .filter(v => v != null);
  if (valid.length === 0) return null;
  if (voting === 'mean') return Math.round(valid.reduce((a, b) => a + b, 0) / valid.length);
  if (voting === 'median') {
    const sorted = [...valid].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? Math.round((sorted[mid - 1] + sorted[mid]) / 2) : sorted[mid];
  }
  if (voting === 'majority') {
    const counts = {};
    valid.forEach(v => { counts[v] = (counts[v] || 0) + 1; });
    const maxCount = Math.max(...Object.values(counts));
    const winners = Object.keys(counts).filter(k => counts[k] === maxCount).map(Number);
    return Math.round(winners.reduce((a, b) => a + b, 0) / winners.length);
  }
  return null;
}

function TrajectoryPlot({ swing, enabledMethods, voting, showEnsemble, phase = 'backswing' }) {
  const data = phase === 'backswing' ? swing.backswing : swing.contact;
  const methodColors = phase === 'backswing' ? METHOD_COLORS : CONTACT_COLORS;
  const methodLabels = phase === 'backswing' ? METHOD_LABELS : CONTACT_LABELS;
  const { wrist_x: wx, wrist_y: wy, production_rel: prodRel, methods } = data;

  if (!wx || wx.length === 0) return null;

  const pad = 20;
  const W = 400, H = 320;
  const xMin = Math.min(...wx), xMax = Math.max(...wx);
  const yMin = Math.min(...wy), yMax = Math.max(...wy);
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
  const scale = Math.min((W - 2 * pad) / xRange, (H - 2 * pad) / yRange);
  const xOff = pad + ((W - 2 * pad) - xRange * scale) / 2;
  const yOff = pad + ((H - 2 * pad) - yRange * scale) / 2;
  const tx = i => xOff + (wx[i] - xMin) * scale;
  const ty = i => yOff + (wy[i] - yMin) * scale;

  const n = wx.length;
  const ensIdx = showEnsemble
    ? computeEnsemble(methods, voting, enabledMethods.filter(m => m !== 'production'))
    : null;

  const pathD = wx.map((_, i) => `${i === 0 ? 'M' : 'L'}${tx(i).toFixed(1)},${ty(i).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 420, background: colors.card, borderRadius: 8, border: `1px solid ${colors.cardBorder}` }}>
      <path d={pathD} fill="none" stroke={colors.textDim} strokeWidth={1} opacity={0.4} />
      {wx.map((_, i) => {
        const t = n > 1 ? i / (n - 1) : 0;
        const r = Math.round(30 + t * 100);
        const g = Math.round(80 + t * 140);
        const b = Math.round(120 + t * 80);
        return <circle key={i} cx={tx(i)} cy={ty(i)} r={3} fill={`rgb(${r},${g},${b})`} opacity={0.7} />;
      })}

      {enabledMethods.includes('production') && prodRel != null && prodRel >= 0 && prodRel < n && (
        <MarkerSvg shape="X" x={tx(prodRel)} y={ty(prodRel)} size={10} color={methodColors.production} />
      )}

      {Object.entries(methods).map(([name, idx]) => {
        if (!enabledMethods.includes(name) || idx == null || idx < 0 || idx >= n) return null;
        return (
          <MarkerSvg key={name} shape={MARKER_SHAPES[name]} x={tx(idx)} y={ty(idx)} size={7} color={methodColors[name]} />
        );
      })}

      {showEnsemble && ensIdx != null && ensIdx >= 0 && ensIdx < n && (
        <MarkerSvg shape="star" x={tx(ensIdx)} y={ty(ensIdx)} size={10} color={colors.green} />
      )}

      <text x={W / 2} y={H - 4} textAnchor="middle" fill={colors.textDim} fontSize={10} fontFamily="DM Sans, sans-serif">
        x (px)
      </text>
      <text x={6} y={H / 2} textAnchor="middle" fill={colors.textDim} fontSize={10} fontFamily="DM Sans, sans-serif" transform={`rotate(-90,6,${H / 2})`}>
        y (px)
      </text>
    </svg>
  );
}

function SignalPlot({ swing, enabledMethods, voting, showEnsemble }) {
  const { frames, smoothed } = swing.signal;
  const { production_rel: prodRel, methods, window_start: wsStart } = swing.backswing;
  const prodFrame = swing.production_frame;

  if (!frames || frames.length === 0) return null;

  const pad = 30;
  const W = 400, H = 200;
  const xMin = frames[0], xMax = frames[frames.length - 1];
  const yMin = Math.min(...smoothed), yMax = Math.max(...smoothed);
  const xRange = xMax - xMin || 1, yRange = yMax - yMin || 1;
  const tx = f => pad + ((f - xMin) / xRange) * (W - 2 * pad);
  const ty = v => pad + ((yMax - v) / yRange) * (H - 2 * pad);

  const pathD = frames.map((f, i) => `${i === 0 ? 'M' : 'L'}${tx(f).toFixed(1)},${ty(smoothed[i]).toFixed(1)}`).join(' ');

  const ensIdx = showEnsemble
    ? computeEnsemble(methods, voting, enabledMethods.filter(m => m !== 'production'))
    : null;
  const ensFrame = ensIdx != null ? wsStart + ensIdx : null;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: 420, background: colors.card, borderRadius: 8, border: `1px solid ${colors.cardBorder}` }}>
      <path d={pathD} fill="none" stroke="#6495ed" strokeWidth={2} />

      {enabledMethods.includes('production') && prodFrame >= xMin && prodFrame <= xMax && (
        <line x1={tx(prodFrame)} y1={pad} x2={tx(prodFrame)} y2={H - pad} stroke={METHOD_COLORS.production} strokeWidth={2} strokeDasharray="6,3" />
      )}

      {Object.entries(methods).map(([name, idx]) => {
        if (!enabledMethods.includes(name) || idx == null) return null;
        const f = wsStart + idx;
        if (f < xMin || f > xMax) return null;
        return (
          <line key={name} x1={tx(f)} y1={pad} x2={tx(f)} y2={H - pad} stroke={METHOD_COLORS[name] || colors.accent} strokeWidth={1.5} strokeDasharray="4,2" />
        );
      })}

      {showEnsemble && ensFrame != null && ensFrame >= xMin && ensFrame <= xMax && (
        <line x1={tx(ensFrame)} y1={pad} x2={tx(ensFrame)} y2={H - pad} stroke={colors.green} strokeWidth={2.5} strokeDasharray="2,2" />
      )}

      <text x={W / 2} y={H - 2} textAnchor="middle" fill={colors.textDim} fontSize={10} fontFamily="DM Sans, sans-serif">
        Frame
      </text>
      <text x={4} y={H / 2} textAnchor="middle" fill={colors.textDim} fontSize={10} fontFamily="DM Sans, sans-serif" transform={`rotate(-90,4,${H / 2})`}>
        Signal
      </text>
    </svg>
  );
}

function ToggleButton({ label, active, color, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: '5px 10px', borderRadius: 6, fontSize: 12, fontWeight: 600,
      fontFamily: "'DM Sans', sans-serif", cursor: 'pointer',
      border: `1.5px solid ${active ? color : colors.cardBorder}`,
      background: active ? `${color}20` : 'transparent',
      color: active ? color : colors.textDim,
      transition: 'all 0.15s ease', outline: 'none',
    }}>
      {label}
    </button>
  );
}

function StatBox({ label, value, color = colors.accent }) {
  return (
    <div style={{
      background: colors.card, border: `1px solid ${colors.cardBorder}`, borderRadius: 8,
      padding: '10px 14px', minWidth: 100, textAlign: 'center',
    }}>
      <div style={{ fontSize: 20, fontWeight: 700, color, fontFamily: "'JetBrains Mono', monospace" }}>{value}</div>
      <div style={{ fontSize: 11, color: colors.textDim, marginTop: 2 }}>{label}</div>
    </div>
  );
}

function DiffTable({ swing, enabledMethods, voting }) {
  const bs = swing.backswing;
  const prod = bs.production_rel;
  const ensIdx = computeEnsemble(bs.methods, voting, enabledMethods.filter(m => m !== 'production'));

  const rows = [];
  if (enabledMethods.includes('production') && prod != null) {
    rows.push({ name: 'Production', frame: bs.window_start + prod, rel: prod, color: METHOD_COLORS.production });
  }
  Object.entries(bs.methods).forEach(([name, idx]) => {
    if (!enabledMethods.includes(name) || idx == null) return;
    rows.push({ name: METHOD_LABELS[name]?.split('—')[0]?.trim() || name, frame: bs.window_start + idx, rel: idx, color: METHOD_COLORS[name] });
  });
  if (ensIdx != null) {
    rows.push({ name: `Ensemble (${VOTING_LABELS[voting]})`, frame: bs.window_start + ensIdx, rel: ensIdx, color: colors.green });
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, fontFamily: "'JetBrains Mono', monospace" }}>
        <thead>
          <tr style={{ borderBottom: `1px solid ${colors.cardBorder}` }}>
            <th style={{ textAlign: 'left', padding: '6px 8px', color: colors.textMuted, fontFamily: "'DM Sans', sans-serif", fontWeight: 600 }}>Method</th>
            <th style={{ textAlign: 'right', padding: '6px 8px', color: colors.textMuted, fontFamily: "'DM Sans', sans-serif", fontWeight: 600 }}>Frame</th>
            <th style={{ textAlign: 'right', padding: '6px 8px', color: colors.textMuted, fontFamily: "'DM Sans', sans-serif", fontWeight: 600 }}>vs Prod</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const diff = prod != null ? r.rel - prod : '-';
            return (
              <tr key={i} style={{ borderBottom: `1px solid ${colors.cardBorder}22` }}>
                <td style={{ padding: '5px 8px', fontFamily: "'DM Sans', sans-serif" }}>
                  <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: r.color, marginRight: 6, verticalAlign: 'middle' }} />
                  {r.name}
                </td>
                <td style={{ textAlign: 'right', padding: '5px 8px', color: colors.text }}>{r.frame}</td>
                <td style={{ textAlign: 'right', padding: '5px 8px', color: diff === 0 || diff === '-' ? colors.textDim : diff > 0 ? colors.amber : colors.accent }}>
                  {diff === '-' ? '-' : (diff > 0 ? `+${diff}` : diff)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const RefinementExplorer = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedSwing, setSelectedSwing] = useState(0);
  const [enabledBS, setEnabledBS] = useState(['production', 'argmin_y', 'argmin_xy', 'velocity_zero']);
  const [enabledCT, setEnabledCT] = useState(['production', 'argmax_y', 'argmax_xy', 'argmax_x']);
  const [voting, setVoting] = useState('mean');
  const [showEnsemble, setShowEnsemble] = useState(true);
  const [filterSession, setFilterSession] = useState('all');
  const [phase, setPhase] = useState('backswing');

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/refinement_data.json`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const sessions = useMemo(() => data ? data.summary.sessions : [], [data]);
  const filteredSwings = useMemo(() => {
    if (!data) return [];
    return filterSession === 'all' ? data.swings : data.swings.filter(s => s.session === filterSession);
  }, [data, filterSession]);

  const swing = filteredSwings[selectedSwing] || null;

  const toggleBS = useCallback(m => setEnabledBS(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]), []);
  const toggleCT = useCallback(m => setEnabledCT(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]), []);

  const enabledMethods = phase === 'backswing' ? enabledBS : enabledCT;
  const methodColors = phase === 'backswing' ? METHOD_COLORS : CONTACT_COLORS;
  const methodLabels = phase === 'backswing' ? METHOD_LABELS : CONTACT_LABELS;
  const toggleMethod = phase === 'backswing' ? toggleBS : toggleCT;

  // Aggregate stats with current settings
  const aggStats = useMemo(() => {
    if (!data) return null;
    const bsMethods = enabledBS.filter(m => m !== 'production');
    let diffs = [];
    data.swings.forEach(s => {
      const prod = s.backswing.production_rel;
      const ens = computeEnsemble(s.backswing.methods, voting, bsMethods);
      if (prod != null && ens != null) diffs.push(Math.abs(prod - ens));
    });
    if (diffs.length === 0) return { mean: 0, agree1: 0, agree3: 0, differ5: 0, n: 0 };
    return {
      mean: (diffs.reduce((a, b) => a + b, 0) / diffs.length).toFixed(1),
      agree1: (diffs.filter(d => d <= 1).length / diffs.length * 100).toFixed(0),
      agree3: (diffs.filter(d => d <= 3).length / diffs.length * 100).toFixed(0),
      differ5: (diffs.filter(d => d >= 5).length / diffs.length * 100).toFixed(0),
      n: diffs.length,
    };
  }, [data, enabledBS, voting]);

  if (loading) {
    return (
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '48px 16px', textAlign: 'center', color: colors.textDim }}>
        Loading refinement data...
      </div>
    );
  }

  if (!data || data.swings.length === 0) {
    return (
      <div style={{ maxWidth: 1100, margin: '0 auto', padding: '48px 16px', textAlign: 'center', color: colors.textDim }}>
        No refinement data found. Run <code>python scripts/generate_refinement_data.py</code> to generate.
      </div>
    );
  }

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div ref={headerRef} style={{
        marginBottom: 24,
        opacity: headerInView ? 1 : 0,
        transform: headerInView ? 'translateY(0)' : 'translateY(16px)',
        transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)',
      }}>
        <h1 style={{
          fontSize: 28, fontWeight: 700, margin: '0 0 6px',
          background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`,
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em',
        }}>Refinement Explorer</h1>
        <p style={{ color: colors.textDim, fontSize: 14, margin: 0 }}>
          Compare 1D production vs 2D ensemble backswing apex detection across {data.summary.total_swings} swings
        </p>
      </div>

      {/* Aggregate Stats */}
      {aggStats && (
        <div style={{ display: 'flex', gap: 12, marginBottom: 20, flexWrap: 'wrap' }}>
          <StatBox label="Total Swings" value={aggStats.n} />
          <StatBox label="Mean |diff|" value={`${aggStats.mean}f`} color={colors.amber} />
          <StatBox label="Agree ≤1f" value={`${aggStats.agree1}%`} color={colors.green} />
          <StatBox label="Differ ≥5f" value={`${aggStats.differ5}%`} color={colors.rose} />
        </div>
      )}

      {/* Controls */}
      <CollapsibleCard title="Controls" sub="methods, voting, filters" icon="⚙" defaultOpen={true}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Phase toggle */}
          <div>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Phase</div>
            <div style={{ display: 'flex', gap: 6 }}>
              <ToggleButton label="Backswing" active={phase === 'backswing'} color={colors.accent} onClick={() => setPhase('backswing')} />
              <ToggleButton label="Contact" active={phase === 'contact'} color={colors.amber} onClick={() => setPhase('contact')} />
            </div>
          </div>

          {/* Method toggles */}
          <div>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Methods</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {Object.entries(methodLabels).map(([key, label]) => (
                <ToggleButton key={key} label={label} active={enabledMethods.includes(key)} color={methodColors[key]} onClick={() => toggleMethod(key)} />
              ))}
            </div>
          </div>

          {/* Voting strategy */}
          <div>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Voting Strategy</div>
            <div style={{ display: 'flex', gap: 6 }}>
              {Object.entries(VOTING_LABELS).map(([key, label]) => (
                <ToggleButton key={key} label={label} active={voting === key} color={colors.green} onClick={() => setVoting(key)} />
              ))}
            </div>
          </div>

          {/* Ensemble toggle */}
          <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
            <ToggleButton label={showEnsemble ? "Ensemble: ON" : "Ensemble: OFF"} active={showEnsemble} color={colors.green} onClick={() => setShowEnsemble(!showEnsemble)} />
          </div>

          {/* Session filter */}
          <div>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Session</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <ToggleButton label="All" active={filterSession === 'all'} color={colors.purple} onClick={() => { setFilterSession('all'); setSelectedSwing(0); }} />
              {sessions.map(s => (
                <ToggleButton key={s} label={s} active={filterSession === s} color={colors.purple} onClick={() => { setFilterSession(s); setSelectedSwing(0); }} />
              ))}
            </div>
          </div>
        </div>
      </CollapsibleCard>

      {/* Swing selector */}
      <div style={{ margin: '16px 0', display: 'flex', alignItems: 'center', gap: 12 }}>
        <button onClick={() => setSelectedSwing(Math.max(0, selectedSwing - 1))} disabled={selectedSwing === 0}
          style={{ padding: '6px 14px', borderRadius: 6, border: `1px solid ${colors.cardBorder}`, background: colors.card, color: colors.text, cursor: 'pointer', fontSize: 14, fontWeight: 700, opacity: selectedSwing === 0 ? 0.3 : 1 }}>
          ←
        </button>
        <div style={{ flex: 1 }}>
          <input type="range" min={0} max={Math.max(0, filteredSwings.length - 1)} value={selectedSwing}
            onChange={e => setSelectedSwing(Number(e.target.value))}
            style={{ width: '100%', accentColor: colors.accent }}
          />
        </div>
        <button onClick={() => setSelectedSwing(Math.min(filteredSwings.length - 1, selectedSwing + 1))} disabled={selectedSwing >= filteredSwings.length - 1}
          style={{ padding: '6px 14px', borderRadius: 6, border: `1px solid ${colors.cardBorder}`, background: colors.card, color: colors.text, cursor: 'pointer', fontSize: 14, fontWeight: 700, opacity: selectedSwing >= filteredSwings.length - 1 ? 0.3 : 1 }}>
          →
        </button>
        <span style={{ color: colors.textMuted, fontSize: 12, fontFamily: "'JetBrains Mono', monospace", minWidth: 80, textAlign: 'right' }}>
          {selectedSwing + 1} / {filteredSwings.length}
        </span>
      </div>

      {/* Swing info */}
      {swing && (
        <div style={{ marginBottom: 16, padding: '8px 12px', background: colors.card, borderRadius: 8, border: `1px solid ${colors.cardBorder}`, display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 13, color: colors.accent }}>{swing.session}/{swing.video}</span>
          <span style={{ color: colors.textDim, fontSize: 12 }}>Swing {swing.swing_idx}</span>
          <span style={{ color: colors.textDim, fontSize: 12 }}>Frame {swing.production_frame}</span>
          {swing.fps && <span style={{ color: colors.textDim, fontSize: 12 }}>{swing.fps}fps</span>}
        </div>
      )}

      {/* Main visualization */}
      {swing && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 16, marginBottom: 20 }}>
          <CollapsibleCard title="2D Wrist Trajectory" sub={phase} icon="📐" defaultOpen={true}>
            <TrajectoryPlot swing={swing} enabledMethods={enabledMethods} voting={voting} showEnsemble={showEnsemble} phase={phase} />
            <div style={{ marginTop: 10 }}>
              <Legend entries={
                [
                  ...(enabledMethods.includes('production') ? [{ color: methodColors.production, shape: MARKER_SHAPES.production, label: 'Production' }] : []),
                  ...Object.keys(phase === 'backswing' ? METHOD_LABELS : CONTACT_LABELS)
                    .filter(k => k !== 'production' && enabledMethods.includes(k))
                    .map(k => ({ color: methodColors[k], shape: MARKER_SHAPES[k], label: (methodLabels[k] || k).split('—')[0].trim() })),
                  ...(showEnsemble ? [{ color: colors.green, shape: 'star', label: `Ensemble (${VOTING_LABELS[voting]})` }] : []),
                ]
              } />
            </div>
          </CollapsibleCard>

          <CollapsibleCard title="1D Combined Signal" sub="smoothed" icon="📈" defaultOpen={true}>
            <SignalPlot swing={swing} enabledMethods={enabledBS} voting={voting} showEnsemble={showEnsemble} />
          </CollapsibleCard>
        </div>
      )}

      {/* Frame comparison table */}
      {swing && (
        <CollapsibleCard title="Frame Picks" sub="per method" icon="📊" defaultOpen={true}>
          <DiffTable swing={swing} enabledMethods={enabledMethods} voting={voting} />
        </CollapsibleCard>
      )}
    </div>
  );
};

function Legend({ entries }) {
  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
      {entries.map((e, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <svg width={16} height={16} viewBox="0 0 16 16">
            <MarkerSvg shape={e.shape} x={8} y={8} size={5} color={e.color} stroke="none" strokeWidth={0} />
          </svg>
          <span style={{ fontSize: 11, color: colors.textMuted }}>{e.label}</span>
        </div>
      ))}
    </div>
  );
}

export default RefinementExplorer;
