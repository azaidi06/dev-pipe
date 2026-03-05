import React from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import Table from '../components/Table';
import StatCard from '../components/StatCard';
import useInView from '../hooks/useInView';

const SubStep = ({ label, desc, color = colors.purple }) => (
  <div style={{ background: `${color}08`, border: `1px solid ${color}20`, borderRadius: '8px', padding: '10px 14px' }}>
    <div style={{ fontSize: '12px', fontWeight: 700, color }}>{label}</div>
    <div style={{ fontSize: '11px', color: colors.textMuted, marginTop: '2px', lineHeight: 1.5 }}>{desc}</div>
  </div>
);

const DeployCosts = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={headerRef} style={{ marginBottom: '32px', opacity: headerInView ? 1 : 0, transform: headerInView ? 'translateY(0)' : 'translateY(16px)', transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>Deploy & Costs</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>AMI baking, spot instances, and cost model</p>
      </div>

      <CollapsibleCard title="Cost Model" sub="Per-video cost breakdown" icon="&#128176;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '20px' }}>
          <StatCard label="Instance" value="g6.2xlarge" sub="spot @ $0.47/hr" color={colors.purple} delay={0} />
          <StatCard label="Per Video" value="~$0.04" sub="5-min video" color={colors.green} delay={80} />
          <StatCard label="Throughput" value="~10/hr" sub="videos per instance" color={colors.accent} delay={160} />
          <StatCard label="Per Frame" value="$0.000003" sub="~18K frames/video" color={colors.amber} delay={240} />
        </div>
        <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
          {[
            { name: 'EC2 spot', pct: 87.5, cost: '$0.035', color: colors.rose },
            { name: 'S3 storage', pct: 7.5, cost: '$0.003', color: colors.accent },
            { name: 'Lambda', pct: 4, cost: '$0.002', color: colors.green },
            { name: 'DynamoDB', pct: 1, cost: '<$0.001', color: colors.amber },
          ].map(item => (
            <div key={item.name} style={{ flex: '1 1 120px', padding: '12px 16px', borderRadius: '10px', background: `${item.color}06`, border: `1px solid ${item.color}15`, textAlign: 'center' }}>
              <div style={{ fontSize: '11px', color: colors.textDim, marginBottom: '4px' }}>{item.name}</div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '16px', fontWeight: 700, color: item.color }}>{item.cost}</div>
              <div style={{ fontSize: '10px', color: colors.textDim, marginTop: '2px' }}>{item.pct}% of total</div>
            </div>
          ))}
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="AMI Stack" sub="Custom-baked with all dependencies" icon="&#128230;">
        <Table
          headers={['Component', 'Version', 'Notes']}
          rows={[
            ['PyTorch', '2.8+cu126', 'cu124 builds don\'t exist for 2.8'],
            ['mmcv-lite', '2.2.0', '4 patches for torch 2.8 compat'],
            ['ffmpeg', 'custom build', 'NVENC + NVDEC support'],
            ['ViTPose-Huge', '\u2014', '~632M params, pre-cached weights'],
            ['RTMDet-M', '\u2014', 'Person detector, pre-cached weights'],
          ]}
        />
      </CollapsibleCard>

      <CollapsibleCard title="Instance Lifecycle" sub="Boot \u2192 process \u2192 auto-terminate" icon="&#128260;">
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
          {[
            { label: 'Launching', desc: 'deploy.sh', color: colors.textDim },
            { label: 'Booting', desc: 'spot fulfilled', color: colors.accent },
            { label: 'Importing', desc: 'user-data runs', color: colors.amber },
            { label: 'Loading', desc: 'Python imports (~91s)', color: colors.rose },
            { label: 'Polling', desc: 'model loaded (~15s)', color: colors.green },
            { label: 'Processing', desc: 'SQS message', color: colors.green },
            { label: 'Idle', desc: 'no messages', color: colors.amber },
            { label: 'Terminated', desc: 'idle timeout', color: colors.textDim },
          ].map((step, i) => (
            <React.Fragment key={step.label}>
              <div style={{ padding: '10px 14px', borderRadius: '10px', background: `${step.color}08`, border: `1px solid ${step.color}20` }}>
                <div style={{ fontSize: '12px', fontWeight: 700, color: step.color }}>{step.label}</div>
                <div style={{ fontSize: '10px', color: colors.textDim, marginTop: '2px' }}>{step.desc}</div>
              </div>
              {i < 7 && <div style={{ display: 'flex', alignItems: 'center' }}><svg width="16" height="12"><polygon points="16,6 8,0 8,12" fill={colors.textDim} opacity="0.3" /></svg></div>}
            </React.Fragment>
          ))}
        </div>
        <p style={{ fontSize: '12px', color: colors.textDim, marginTop: '14px' }}>Auto-terminate on idle prevents paying for GPU time when the queue is empty.</p>
      </CollapsibleCard>
    </div>
  );
};

export default DeployCosts;
