import React from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import Table from '../components/Table';
import CodeBlock from '../components/CodeBlock';
import StatCard from '../components/StatCard';
import useInView from '../hooks/useInView';

const BarItem = ({ label, value, maxValue, color }) => {
  const pct = (value / maxValue) * 100;
  return (
    <div style={{ marginBottom: '10px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
        <span style={{ color: colors.textMuted }}>{label}</span>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color }}>{value}s</span>
      </div>
      <div style={{ width: '100%', height: '8px', background: `${color}15`, borderRadius: '4px', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: `linear-gradient(90deg, ${color}, ${color}cc)`, borderRadius: '4px', transition: 'width 0.8s cubic-bezier(0.16, 1, 0.3, 1)', boxShadow: `0 0 12px ${color}40` }} />
      </div>
    </div>
  );
};

const PipelineFlow = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });

  const timingData = [
    { label: 'S3 download', value: 15, color: colors.textDim },
    { label: 'Transcode (NVENC)', value: 74, color: colors.amber },
    { label: 'Labeling (turbo)', value: 313, color: colors.rose },
    { label: 'S3 upload', value: 35, color: colors.textDim },
    { label: 'Swing detect', value: 13, color: colors.accent },
    { label: 'Post process', value: 5, color: colors.accent },
  ];

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={headerRef} style={{
        marginBottom: '32px', opacity: headerInView ? 1 : 0,
        transform: headerInView ? 'translateY(0)' : 'translateY(16px)',
        transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)',
      }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>Pipeline Flow</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>How data moves from phone to analysis</p>
      </div>

      <CollapsibleCard title="S3 Layout" sub="All data in golf-swing-data with golfer-first keys" icon="&#128193;">
        <CodeBlock>{`{golfer}/
\u251c\u2500\u2500 raw/          # Original .MOV uploads (iPhone HEVC 10-bit VFR)
\u251c\u2500\u2500 processed/    # Transcoded .mp4 (H.264 CFR yuv420p)
\u251c\u2500\u2500 keypoints/    # ViTPose .pkl files (17 keypoints per frame)
\u251c\u2500\u2500 detection/    # Swing detection .json (backswing + contact frames)
\u251c\u2500\u2500 fingers/      # Finger prediction .json
\u251c\u2500\u2500 frames/       # Extracted JPGs with skeleton overlay
\u251c\u2500\u2500 output/       # Grids, signal plots, hand crops
\u2514\u2500\u2500 analysis/     # Biomechanical comparison plots + AI text`}</CodeBlock>
      </CollapsibleCard>

      <CollapsibleCard title="Inter-Stage Messaging" sub="S3 event notifications, no orchestrator" icon="&#128172;">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {[
            { from: 'Phone', action: 'Upload .MOV to /raw/', color: colors.green },
            { from: 'S3', action: 'S3 event (suffix .MOV/.mp4) \u2192 SQS', color: colors.green },
            { from: 'EC2', action: 'worker.py polls SQS, uploads .mp4 + .pkl', color: colors.purple },
            { from: 'S3', action: 'S3 event (suffix .pkl) \u2192 swing_detection Lambda', color: colors.amber },
            { from: 'Lambda', action: 'Uploads .json to /detection/', color: colors.amber },
            { from: 'S3', action: 'S3 event (suffix .json) \u2192 post_processing Lambda', color: colors.amber },
            { from: 'Lambda', action: 'Uploads to /fingers/, /frames/, /output/', color: colors.amber },
          ].map((step, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '10px 16px', borderRadius: '10px', background: `${step.color}06`, border: `1px solid ${step.color}15` }}>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '11px', fontWeight: 700, color: step.color, minWidth: '60px' }}>{step.from}</span>
              <svg width="16" height="12"><polygon points="16,6 8,0 8,12" fill={step.color} opacity="0.5" /></svg>
              <span style={{ fontSize: '12px', color: colors.textMuted }}>{step.action}</span>
            </div>
          ))}
        </div>
        <p style={{ fontSize: '12px', color: colors.textDim, marginTop: '14px', lineHeight: 1.6 }}>
          Each stage fires-and-forgets to S3, and the next stage picks it up via event triggers. No central coordinator.
        </p>
      </CollapsibleCard>

      <CollapsibleCard title="Timing Breakdown" sub="5-min 60fps iPhone video (~18,000 frames)" icon="&#9201;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }}>
          <div>
            {timingData.map(item => (
              <BarItem key={item.label} {...item} maxValue={313} />
            ))}
          </div>
          <div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '16px' }}>
              {[
                { color: colors.amber, label: 'GPU hardware (NVENC/NVDEC)' },
                { color: colors.rose, label: 'GPU compute (CUDA, bottleneck)' },
                { color: colors.textDim, label: 'Network I/O' },
                { color: colors.accent, label: 'Lambda CPU' },
              ].map(item => (
                <div key={item.label} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: item.color }} />
                  <span style={{ fontSize: '11px', color: colors.textMuted }}>{item.label}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="Cold Start" sub="First video on a fresh instance" icon="&#10052;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '16px' }}>
          <StatCard label="Boot + user-data" value="20s" sub="cloud-init, S3 script pull" color={colors.textDim} delay={0} />
          <StatCard label="Python imports" value="91s" sub="EBS lazy-loading (~12 MB/s)" color={colors.rose} delay={80} />
          <StatCard label="Model load" value="15s" sub="Local checkpoint files" color={colors.purple} delay={160} />
          <StatCard label="Total" value="~2.1 min" sub="Amortized over session" color={colors.accent} delay={240} />
        </div>
      </CollapsibleCard>
    </div>
  );
};

export default PipelineFlow;
