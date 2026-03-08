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

function cropPhaseData(phaseData, before, after, direction) {
  const { wrist_x, wrist_y, production_rel: prodRel, methods } = phaseData;
  if (!wrist_x || wrist_x.length === 0 || prodRel == null) return phaseData;

  const cropStart = Math.max(0, prodRel - before);
  const cropEnd = Math.min(wrist_x.length, prodRel + after + 1);
  const cx = wrist_x.slice(cropStart, cropEnd);
  const cy = wrist_y.slice(cropStart, cropEnd);
  const newProd = prodRel - cropStart;

  // Recompute geometric methods on cropped window
  const newMethods = {};
  if (direction === 'backswing') {
    newMethods.argmin_y = argminArr(cy);
    newMethods.argmin_xy = argminSumArr(cx, cy);
    // velocity_zero: keep original if in range, else null
    const vzOrig = methods.velocity_zero;
    newMethods.velocity_zero = (vzOrig != null && vzOrig >= cropStart && vzOrig < cropEnd) ? vzOrig - cropStart : null;
  } else {
    newMethods.argmax_y = argmaxArr(cy);
    newMethods.argmax_xy = argmaxSumArr(cx, cy);
    newMethods.argmax_x = argmaxArr(cx);
  }

  return {
    ...phaseData,
    wrist_x: cx,
    wrist_y: cy,
    production_rel: newProd,
    methods: newMethods,
    window_start: phaseData.window_start + cropStart,
    window_end: phaseData.window_start + cropEnd,
  };
}

function argminArr(arr) {
  let mi = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] < arr[mi]) mi = i;
  return mi;
}
function argmaxArr(arr) {
  let mi = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[mi]) mi = i;
  return mi;
}
function argminSumArr(ax, ay) {
  let mi = 0, mv = ax[0] + ay[0];
  for (let i = 1; i < ax.length; i++) { const v = ax[i] + ay[i]; if (v < mv) { mv = v; mi = i; } }
  return mi;
}
function argmaxSumArr(ax, ay) {
  let mi = 0, mv = ax[0] + ay[0];
  for (let i = 1; i < ax.length; i++) { const v = ax[i] + ay[i]; if (v > mv) { mv = v; mi = i; } }
  return mi;
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

function computeCurvature(wx, wy) {
  // Discrete curvature: angle change between consecutive segments
  const curv = new Float64Array(wx.length);
  for (let i = 1; i < wx.length - 1; i++) {
    const ax = wx[i] - wx[i - 1], ay = wy[i] - wy[i - 1];
    const bx = wx[i + 1] - wx[i], by = wy[i + 1] - wy[i];
    const dot = ax * bx + ay * by;
    const cross = ax * by - ay * bx;
    curv[i] = Math.abs(Math.atan2(cross, dot));
  }
  return curv;
}

function getPickIndices(methods, enabledMethods, prodRel, ensIdx) {
  const picks = [];
  if (enabledMethods.includes('production') && prodRel != null && prodRel >= 0) picks.push(prodRel);
  Object.entries(methods).forEach(([name, idx]) => {
    if (enabledMethods.includes(name) && idx != null && idx >= 0) picks.push(idx);
  });
  if (ensIdx != null && ensIdx >= 0) picks.push(ensIdx);
  return picks;
}

function renderTrajectory(wx, wy, txFn, tyFn, n, curv, maxCurv, methods, enabledMethods, methodColors, prodRel, ensIdx, showEnsemble, dotScale = 1, markerScale = 1) {
  const pathD = wx.map((_, i) => `${i === 0 ? 'M' : 'L'}${txFn(i).toFixed(1)},${tyFn(i).toFixed(1)}`).join(' ');
  return (
    <>
      <path d={pathD} fill="none" stroke={colors.textDim} strokeWidth={1} opacity={0.4} />
      {wx.map((_, i) => {
        const t = n > 1 ? i / (n - 1) : 0;
        // Green (early) → Yellow (mid) → Red (late)
        const r = Math.round(t < 0.5 ? t * 2 * 220 : 220);
        const g = Math.round(t < 0.5 ? 180 : 180 * (1 - (t - 0.5) * 2));
        const b = Math.round(30 * (1 - t));
        const c = maxCurv > 0 ? curv[i] / maxCurv : 0;
        const radius = (1.5 + c * 5) * dotScale;
        const opacity = 0.5 + c * 0.5;
        return <circle key={i} cx={txFn(i)} cy={tyFn(i)} r={radius} fill={`rgb(${r},${g},${b})`} opacity={opacity} />;
      })}
      {enabledMethods.includes('production') && prodRel != null && prodRel >= 0 && prodRel < n && (
        <MarkerSvg shape="X" x={txFn(prodRel)} y={tyFn(prodRel)} size={10 * markerScale} color={methodColors.production} />
      )}
      {Object.entries(methods).map(([name, idx]) => {
        if (!enabledMethods.includes(name) || idx == null || idx < 0 || idx >= n) return null;
        return <MarkerSvg key={name} shape={MARKER_SHAPES[name]} x={txFn(idx)} y={tyFn(idx)} size={7 * markerScale} color={methodColors[name]} />;
      })}
      {showEnsemble && ensIdx != null && ensIdx >= 0 && ensIdx < n && (
        <MarkerSvg shape="star" x={txFn(ensIdx)} y={tyFn(ensIdx)} size={10 * markerScale} color={colors.green} />
      )}
    </>
  );
}

function TrajectoryPlot({ swing, enabledMethods, voting, showEnsemble, phase = 'backswing' }) {
  const data = phase === 'backswing' ? swing.backswing : swing.contact;
  const methodColors = phase === 'backswing' ? METHOD_COLORS : CONTACT_COLORS;
  const { wrist_x: wx, wrist_y: wy, production_rel: prodRel, methods } = data;

  if (!wx || wx.length === 0) return null;

  const n = wx.length;
  const curv = computeCurvature(wx, wy);
  const maxCurv = Math.max(...curv) || 1;
  const ensIdx = showEnsemble
    ? computeEnsemble(methods, voting, enabledMethods.filter(m => m !== 'production'))
    : null;

  // --- Overview (full trajectory, compact) ---
  const oW = 500, oH = 160, oPad = 16;
  const oxMin = Math.min(...wx), oxMax = Math.max(...wx);
  const oyMin = Math.min(...wy), oyMax = Math.max(...wy);
  const oxR = oxMax - oxMin || 1, oyR = oyMax - oyMin || 1;
  const oScale = Math.min((oW - 2 * oPad) / oxR, (oH - 2 * oPad) / oyR);
  const oxOff = oPad + ((oW - 2 * oPad) - oxR * oScale) / 2;
  const oyOff = oPad + ((oH - 2 * oPad) - oyR * oScale) / 2;
  const otx = i => oxOff + (wx[i] - oxMin) * oScale;
  const oty = i => oyOff + (wy[i] - oyMin) * oScale;

  // --- Zoomed inset (around method picks) ---
  const zW = 500, zH = 380, zPad = 28;
  const picks = getPickIndices(methods, enabledMethods, prodRel, ensIdx);
  let zxMin, zxMax, zyMin, zyMax;
  if (picks.length > 0) {
    const px = picks.filter(p => p < n).map(p => wx[p]);
    const py = picks.filter(p => p < n).map(p => wy[p]);
    const cxMin = Math.min(...px), cxMax = Math.max(...px);
    const cyMin = Math.min(...py), cyMax = Math.max(...py);
    // Expand by 3x the pick spread (or minimum 5% of full range) for context
    const xSpread = Math.max(cxMax - cxMin, oxR * 0.05);
    const ySpread = Math.max(cyMax - cyMin, oyR * 0.05);
    const margin = 2.0;
    const cx = (cxMin + cxMax) / 2, cy = (cyMin + cyMax) / 2;
    zxMin = cx - xSpread * margin; zxMax = cx + xSpread * margin;
    zyMin = cy - ySpread * margin; zyMax = cy + ySpread * margin;
  } else {
    zxMin = oxMin; zxMax = oxMax; zyMin = oyMin; zyMax = oyMax;
  }
  const zxR = zxMax - zxMin || 1, zyR = zyMax - zyMin || 1;
  const zScale = Math.min((zW - 2 * zPad) / zxR, (zH - 2 * zPad) / zyR);
  const zxOff = zPad + ((zW - 2 * zPad) - zxR * zScale) / 2;
  const zyOff = zPad + ((zH - 2 * zPad) - zyR * zScale) / 2;
  const ztx = i => zxOff + (wx[i] - zxMin) * zScale;
  const zty = i => zyOff + (wy[i] - zyMin) * zScale;

  // Zoom box coords in overview space
  const zBoxL = oxOff + (zxMin - oxMin) * oScale;
  const zBoxT = oyOff + (zyMin - oyMin) * oScale;
  const zBoxW = zxR * oScale;
  const zBoxH = zyR * oScale;

  return (
    <div>
      {/* Zoomed inset — apex region */}
      <svg viewBox={`0 0 ${zW} ${zH}`} style={{ width: '100%', background: colors.card, borderRadius: 8, border: `1px solid ${colors.accent}30` }}>
        <text x={zW - 8} y={16} textAnchor="end" fill={colors.accent} fontSize={10} fontFamily="JetBrains Mono, monospace" opacity={0.6}>APEX ZOOM</text>
        {renderTrajectory(wx, wy, ztx, zty, n, curv, maxCurv, methods, enabledMethods, methodColors, prodRel, ensIdx, showEnsemble, 1.5, 1.4)}
        <text x={zW / 2} y={zH - 4} textAnchor="middle" fill={colors.textDim} fontSize={10} fontFamily="DM Sans, sans-serif">x (px)</text>
        <text x={6} y={zH / 2} textAnchor="middle" fill={colors.textDim} fontSize={10} fontFamily="DM Sans, sans-serif" transform={`rotate(-90,6,${zH / 2})`}>y (px)</text>
      </svg>

      {/* Overview — full trajectory, compact */}
      <svg viewBox={`0 0 ${oW} ${oH}`} style={{ width: '100%', marginTop: 8, background: colors.card, borderRadius: 8, border: `1px solid ${colors.cardBorder}` }}>
        <text x={oW - 8} y={14} textAnchor="end" fill={colors.textDim} fontSize={9} fontFamily="JetBrains Mono, monospace" opacity={0.5}>OVERVIEW</text>
        {renderTrajectory(wx, wy, otx, oty, n, curv, maxCurv, methods, enabledMethods, methodColors, prodRel, ensIdx, showEnsemble, 0.7, 0.8)}
        {/* Zoom box indicator */}
        <rect x={zBoxL} y={zBoxT} width={Math.max(zBoxW, 4)} height={Math.max(zBoxH, 4)}
          fill="none" stroke={colors.accent} strokeWidth={1.5} strokeDasharray="4,2" opacity={0.7} rx={2} />
      </svg>
    </div>
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
  const [enabledSessions, setEnabledSessions] = useState(null); // null = all
  const [phase, setPhase] = useState('backswing');
  const [winBefore, setWinBefore] = useState(8);
  const [winAfter, setWinAfter] = useState(8);
  const [minDiff, setMinDiff] = useState(0);
  const [validatedOnly, setValidatedOnly] = useState(true);

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/refinement_data.json`)
      .then(r => r.json())
      .then(d => { setData(d); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  const sessions = useMemo(() => data ? data.summary.sessions : [], [data]);
  const filteredSwings = useMemo(() => {
    if (!data) return [];
    const bsMethods = enabledBS.filter(m => m !== 'production');
    let swings = enabledSessions === null ? data.swings : data.swings.filter(s => enabledSessions.includes(s.session));
    if (validatedOnly) {
      swings = swings.filter(s => s.validated);
    }
    if (minDiff > 0) {
      swings = swings.filter(s => {
        const cropped = cropPhaseData(s.backswing, winBefore, winAfter, 'backswing');
        const prod = cropped.production_rel;
        const ens = computeEnsemble(cropped.methods, voting, bsMethods);
        if (prod == null || ens == null) return false;
        return Math.abs(prod - ens) >= minDiff;
      });
    }
    return swings;
  }, [data, enabledSessions, minDiff, validatedOnly, enabledBS, voting, winBefore, winAfter]);

  // Reset selection when filters change
  const swing = filteredSwings[Math.min(selectedSwing, Math.max(0, filteredSwings.length - 1))] || null;

  const toggleBS = useCallback(m => setEnabledBS(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]), []);
  const toggleCT = useCallback(m => setEnabledCT(prev => prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]), []);

  // Crop swing data based on window sliders
  const croppedSwing = useMemo(() => {
    if (!swing) return null;
    const dir = phase === 'backswing' ? 'backswing' : 'contact';
    const croppedPhase = cropPhaseData(swing[dir], winBefore, winAfter, dir);
    return { ...swing, [dir]: croppedPhase };
  }, [swing, winBefore, winAfter, phase]);

  // Max window bounds for current swing
  const winMax = useMemo(() => {
    if (!swing) return { before: 40, after: 50 };
    const d = phase === 'backswing' ? swing.backswing : swing.contact;
    const prod = d.production_rel ?? 0;
    return { before: prod, after: (d.wrist_x?.length ?? 0) - prod - 1 };
  }, [swing, phase]);

  const enabledMethods = phase === 'backswing' ? enabledBS : enabledCT;
  const methodColors = phase === 'backswing' ? METHOD_COLORS : CONTACT_COLORS;
  const methodLabels = phase === 'backswing' ? METHOD_LABELS : CONTACT_LABELS;
  const toggleMethod = phase === 'backswing' ? toggleBS : toggleCT;

  // Aggregate stats with current settings (using cropped window)
  const aggStats = useMemo(() => {
    if (!data) return null;
    const bsMethods = enabledBS.filter(m => m !== 'production');
    let diffs = [];
    let pool = enabledSessions === null ? data.swings : data.swings.filter(s => enabledSessions.includes(s.session));
    if (validatedOnly) pool = pool.filter(s => s.validated);
    pool.forEach(s => {
      const cropped = cropPhaseData(s.backswing, winBefore, winAfter, 'backswing');
      const prod = cropped.production_rel;
      const ens = computeEnsemble(cropped.methods, voting, bsMethods);
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
  }, [data, enabledBS, voting, winBefore, winAfter, validatedOnly, enabledSessions]);

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

          {/* Window size sliders */}
          <div>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Window Around Peak</div>
            <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 180 }}>
                <span style={{ fontSize: 12, color: colors.textMuted, minWidth: 50 }}>Before</span>
                <input type="range" min={2} max={winMax.before} value={Math.min(winBefore, winMax.before)}
                  onChange={e => setWinBefore(Number(e.target.value))}
                  style={{ flex: 1, accentColor: colors.accent }} />
                <span style={{ fontSize: 12, color: colors.accent, fontFamily: "'JetBrains Mono', monospace", minWidth: 28, textAlign: 'right' }}>{Math.min(winBefore, winMax.before)}f</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 180 }}>
                <span style={{ fontSize: 12, color: colors.textMuted, minWidth: 50 }}>After</span>
                <input type="range" min={2} max={winMax.after} value={Math.min(winAfter, winMax.after)}
                  onChange={e => setWinAfter(Number(e.target.value))}
                  style={{ flex: 1, accentColor: colors.accent }} />
                <span style={{ fontSize: 12, color: colors.accent, fontFamily: "'JetBrains Mono', monospace", minWidth: 28, textAlign: 'right' }}>{Math.min(winAfter, winMax.after)}f</span>
              </div>
            </div>
          </div>

          {/* Score-hand validation filter */}
          <div>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Swing Validation</div>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
              <ToggleButton label={validatedOnly ? "Score-Hand Validated Only" : "All Detections (incl. non-swings)"} active={validatedOnly} color={colors.green}
                onClick={() => { setValidatedOnly(!validatedOnly); setSelectedSwing(0); }} />
            </div>
          </div>

          {/* Min difference filter */}
          <div>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Min |diff| (prod vs ensemble)</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {[0, 1, 2, 3, 4, 5, 8, 10].map(v => (
                <ToggleButton key={v} label={v === 0 ? 'All' : `≥${v}f`} active={minDiff === v} color={colors.rose}
                  onClick={() => { setMinDiff(v); setSelectedSwing(0); }} />
              ))}
            </div>
          </div>

          {/* Session filter (multi-select) */}
          <div>
            <div style={{ fontSize: 11, color: colors.textDim, marginBottom: 6, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Sessions</div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              <ToggleButton label="All" active={enabledSessions === null} color={colors.purple}
                onClick={() => { setEnabledSessions(null); setSelectedSwing(0); }} />
              {sessions.map(s => {
                const active = enabledSessions === null || enabledSessions.includes(s);
                return (
                  <ToggleButton key={s} label={s} active={active} color={colors.purple}
                    onClick={() => {
                      setSelectedSwing(0);
                      if (enabledSessions === null) {
                        // Switching from "all" → deselect this one session
                        setEnabledSessions(sessions.filter(x => x !== s));
                      } else if (active) {
                        const next = enabledSessions.filter(x => x !== s);
                        setEnabledSessions(next.length === 0 ? null : next);
                      } else {
                        const next = [...enabledSessions, s];
                        setEnabledSessions(next.length === sessions.length ? null : next);
                      }
                    }} />
                );
              })}
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
      {croppedSwing && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginBottom: 20 }}>
          <CollapsibleCard title="2D Wrist Trajectory" sub={`${phase} — zoomed apex + overview`} icon="📐" defaultOpen={true}>
            <TrajectoryPlot swing={croppedSwing} enabledMethods={enabledMethods} voting={voting} showEnsemble={showEnsemble} phase={phase} />
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
            <SignalPlot swing={croppedSwing} enabledMethods={enabledBS} voting={voting} showEnsemble={showEnsemble} />
          </CollapsibleCard>
        </div>
      )}

      {/* Frame comparison table */}
      {croppedSwing && (
        <CollapsibleCard title="Frame Picks" sub="per method" icon="📊" defaultOpen={true}>
          <DiffTable swing={croppedSwing} enabledMethods={enabledMethods} voting={voting} />
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
