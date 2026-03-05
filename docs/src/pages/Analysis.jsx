import React from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import CodeBlock from '../components/CodeBlock';
import useInView from '../hooks/useInView';

const Analysis = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });

  const metrics = {
    Rotation: ['Shoulder Turn (Proxy)', 'Hip Turn (Proxy)', 'X-Factor (Stretch)', 'Shoulder Rot. Vel.', 'Hip Rot. Vel.'],
    Posture: ['Spine Angle', 'Shoulder Tilt', 'Hip Tilt', 'Lead Knee Angle', 'Trail Knee Angle', 'Right Elbow Angle'],
    Linear: ['Hip Sway (X)', 'Head Sway (X)', 'Head Height (Y)', 'Hand Height (Y)', 'Hand Depth (X)', 'Hand-Body Dist'],
  };

  const metricColors = { Rotation: colors.accent, Posture: colors.purple, Linear: colors.green };

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={headerRef} style={{ marginBottom: '32px', opacity: headerInView ? 1 : 0, transform: headerInView ? 'translateY(0)' : 'translateY(16px)', transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>Biomechanical Analysis</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>Swing comparison, metrics, and statistical analysis</p>
      </div>

      <CollapsibleCard title="Overview" sub="Keypoint pkls + detection results \u2192 analysis" icon="&#128202;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '16px' }}>
          {[
            { title: 'Metric Extraction', desc: 'Shoulder turn, hip rotation, X-factor, spine angle', color: colors.accent },
            { title: 'SPM Comparison', desc: 'Statistical Parametric Mapping between swing groups', color: colors.purple },
            { title: 'Medoid Selection', desc: 'Most representative swing from a set', color: colors.green },
            { title: 'Visualization', desc: 'Overlay plots, metric time series, comparison grids', color: colors.amber },
          ].map(item => (
            <div key={item.title} style={{ padding: '14px 16px', borderRadius: '10px', background: `${item.color}06`, border: `1px solid ${item.color}15` }}>
              <div style={{ fontSize: '12px', fontWeight: 700, color: item.color, marginBottom: '4px' }}>{item.title}</div>
              <div style={{ fontSize: '11px', color: colors.textMuted, lineHeight: 1.5 }}>{item.desc}</div>
            </div>
          ))}
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="Metrics Computed" sub="Per-frame across swing phases" icon="&#128207;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '16px' }}>
          {Object.entries(metrics).map(([category, items]) => (
            <div key={category} style={{ padding: '16px', borderRadius: '12px', background: `${metricColors[category]}06`, border: `1px solid ${metricColors[category]}15` }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: metricColors[category], marginBottom: '10px' }}>{category}</div>
              {items.map(m => (
                <div key={m} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                  <div style={{ width: '5px', height: '5px', borderRadius: '50%', background: metricColors[category] }} />
                  <span style={{ fontSize: '12px', color: colors.textMuted }}>{m}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="Usage" icon="&#128187;">
        <CodeBlock title="Compute metrics from keypoints">{`from eagle_swing.analysis.core import compute_swing_metrics
from eagle_swing.data.loader import load_pkl, interpolate_and_smooth

kd = load_pkl("video.pkl")
smoothed = interpolate_and_smooth(kd)
metrics = compute_swing_metrics(smoothed.keypoints)
print(metrics.keys())
# dict_keys(['Hand Depth (X)', 'Hand Height (Y)', 'Spine Angle', ...])`}</CodeBlock>
        <CodeBlock title="SPM comparison between groups">{`from eagle_swing.analysis.core import perform_spm_1d

t_stat, sig_mask, p_vals = perform_spm_1d(group_a_metrics, group_b_metrics)
# sig_mask: boolean array, True where groups differ significantly`}</CodeBlock>
        <CodeBlock title="CLI pipeline">{`python scripts/run_analysis.py --golfer ymirza --day oct25 --phases backswing downswing`}</CodeBlock>
      </CollapsibleCard>

      <CollapsibleCard title="Analysis Pipeline Flow" sub="From CSV index to results" icon="&#128260;">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
          {[
            { step: 'CSV index', desc: 'golfer, day, pkls', color: colors.textDim },
            { step: 'Load pkl + detection', desc: 'Per swing', color: colors.accent },
            { step: 'Phase extraction', desc: 'Backswing / downswing', color: colors.accent },
            { step: 'Time normalization', desc: '0-100% of phase', color: colors.green },
            { step: 'Metric computation', desc: '17 biomechanical metrics', color: colors.green },
            { step: 'Medoid selection', desc: 'Most representative swing', color: colors.purple },
            { step: 'SPM t-test', desc: 'Group comparison', color: colors.purple },
            { step: 'Visualization', desc: 'Comparison plots', color: colors.amber },
          ].map((item, i) => (
            <div key={item.step} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '8px 14px', borderRadius: '8px', background: `${item.color}06`, border: `1px solid ${item.color}12` }}>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '10px', fontWeight: 700, color: item.color, minWidth: '20px' }}>{i + 1}</span>
              <span style={{ fontSize: '12px', fontWeight: 600, color: colors.text, minWidth: '140px' }}>{item.step}</span>
              <span style={{ fontSize: '11px', color: colors.textDim }}>{item.desc}</span>
            </div>
          ))}
        </div>
      </CollapsibleCard>
    </div>
  );
};

export default Analysis;
