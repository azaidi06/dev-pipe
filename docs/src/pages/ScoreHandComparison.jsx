import React from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import Table from '../components/Table';
import CodeBlock from '../components/CodeBlock';
import StatCard from '../components/StatCard';
import Badge from '../components/Badge';
import useInView from '../hooks/useInView';

const ScoreHandComparison = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });

  const sessionData = [
    ['nov16', '42', '3', '42/42 (100%)', '0.40', '1', '100%', '0%'],
    ['oct25', '34', '6', '34/34 (100%)', '0.32', '1', '100%', '0%'],
    ['sep14', '47', '8', '45/47 (96%)', '43.64', '1026', '96%', '4%'],
    ['aug9', '64', '6', '57/64 (89%)', '14.28', '262', '80%', '20%'],
    ['jun8', '85', '13', '45/85 (53%)', '189.66', '1093', '0%', '100%'],
  ];
  const totalRow = ['ALL', '272', '36', '223/272 (82%)', '70.27', '1093', '63%', '37%'];
  totalRow.bold = true;

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={headerRef} style={{ marginBottom: '32px', opacity: headerInView ? 1 : 0, transform: headerInView ? 'translateY(0)' : 'translateY(16px)', transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>Score-Hand Comparison</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>2D anatomical ensemble vs production 1D pipeline for backswing detection</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '24px' }}>
        <StatCard label="Sequences" value="272" sub="across 36 videos" color={colors.accent} delay={0} />
        <StatCard label="Co-detected" value="82%" sub="223 / 272 matched" color={colors.green} delay={100} />
        <StatCard label="Perfect match" value="63%" sub="|diff| < 2 frames" color={colors.purple} delay={200} />
        <StatCard label="Sessions" value="5" sub="jun8 \u2192 nov16" color={colors.amber} delay={300} />
      </div>

      <CollapsibleCard title="Per-Session Results" sub="Co-detection rate and frame agreement" icon="&#128202;" badge="5 sessions">
        <Table
          headers={['Session', 'Seqs', 'Videos', 'Co-detected', 'Mean |diff|', 'Max |diff|', '|diff| < 2', '|diff| \u2265 3']}
          rows={[...sessionData, totalRow]}
          highlightCol={3}
        />
        <p style={{ fontSize: '12px', color: colors.textDim, marginTop: '4px', lineHeight: 1.6 }}>
          <strong style={{ color: colors.green }}>Co-detected</strong> = both pipelines found the same swing within 30 frames. <strong style={{ color: colors.text }}>|diff| {'<'} 2</strong> = score-hand ensemble and production agree within 1 frame.
        </p>
      </CollapsibleCard>

      <CollapsibleCard title="Why 2D Beats 1D" sub="The wrist traces an arc, not a line" icon="&#128270;">
        <p style={{ fontSize: '13px', color: colors.textMuted, lineHeight: 1.6, marginBottom: '16px' }}>
          During a backswing, the wrist traces a 2D arc. The highest vertical point (min y) and the furthest-back point (min x) typically occur at <strong style={{ color: colors.text }}>different frames</strong>. The production pipeline's combined 1D signal conflates these two dimensions.
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
          {[
            { method: 'argmin(y)', desc: 'Highest vertical point of the wrist', color: colors.amber },
            { method: 'argmin(x+y)', desc: 'Top-left corner of the wrist arc', color: colors.purple },
            { method: 'velocity zero-crossing', desc: 'Where vertical motion reverses direction', color: colors.rose },
          ].map(m => (
            <div key={m.method} style={{ padding: '14px 16px', borderRadius: '10px', background: `${m.color}06`, border: `1px solid ${m.color}15` }}>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', fontWeight: 700, color: m.color, marginBottom: '4px' }}>{m.method}</div>
              <div style={{ fontSize: '11px', color: colors.textMuted }}>{m.desc}</div>
            </div>
          ))}
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="Pipeline Summary" sub="Complete flow from anchor to landmarks" icon="&#128736;">
        <CodeBlock>{`find_score_hand(kpe)  \u2192  follow-through anchor (per swing)
    \u2502
    \u251c\u2500 backward: find_all_higher_wrist_idxs(kps)  \u2192  backswing clusters
    \u2502   \u2514\u2500 match nearest preceding cluster
    \u2502       \u2514\u2500 [bs_anchor \u2212 10, bs_anchor + 20]  \u2192  2D ensemble  \u2192  backswing top
    \u2502
    \u2514\u2500 forward: [bs_top + 10, bs_top + 40]  \u2192  inverse ensemble  \u2192  contact point`}</CodeBlock>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '8px' }}>
          <Badge color={colors.green}>find_score_hand</Badge>
          <Badge color={colors.accent}>find_all_higher_wrist_idxs</Badge>
          <Badge color={colors.rose}>2D backswing ensemble</Badge>
          <Badge color={colors.amber}>2D contact ensemble</Badge>
        </div>
      </CollapsibleCard>
    </div>
  );
};

export default ScoreHandComparison;
