import React from 'react';
import { colors, serviceColors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import StatCard from '../components/StatCard';
import Table from '../components/Table';
import CodeBlock from '../components/CodeBlock';
import useInView from '../hooks/useInView';

const FlowArrow = ({ label, color = colors.textDim }) => (
  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '4px 0' }}>
    <svg width="2" height="20"><line x1="1" y1="0" x2="1" y2="20" stroke={color} strokeWidth="2" /></svg>
    {label && <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '10px', fontWeight: 500, color: colors.textMuted, background: `${color}15`, padding: '2px 10px', borderRadius: '10px', margin: '2px 0', whiteSpace: 'nowrap' }}>{label}</span>}
    <svg width="12" height="8"><polygon points="6,8 0,0 12,0" fill={color} /></svg>
  </div>
);

const FlowNode = ({ title, subtitle, type = 'default', details }) => {
  const borderColor = serviceColors[type] || serviceColors.default;
  return (
    <div style={{
      background: colors.card, border: `1px solid ${colors.cardBorder}`,
      borderLeft: `3px solid ${borderColor}`, borderRadius: '10px', padding: '14px 18px',
      width: '100%', maxWidth: '420px',
    }}>
      <div style={{ fontSize: '14px', fontWeight: 700, color: colors.text }}>{title}</div>
      {subtitle && <div style={{ fontSize: '12px', fontWeight: 600, color: colors.textMuted, marginTop: '2px' }}>{subtitle}</div>}
      {details && <div style={{ fontSize: '11px', color: colors.textDim, marginTop: '4px', lineHeight: 1.5 }}>{details}</div>}
    </div>
  );
};

const Overview = () => {
  const [heroRef, heroInView] = useInView({ threshold: 0.1 });

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={heroRef} style={{
        textAlign: 'center', marginBottom: '40px',
        opacity: heroInView ? 1 : 0, transform: heroInView ? 'translateY(0)' : 'translateY(16px)',
        transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)',
      }}>
        <div style={{ width: '48px', height: '48px', borderRadius: '14px', margin: '0 auto 14px', background: `linear-gradient(135deg, ${colors.accent}, ${colors.green})`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '24px' }}>&#9971;</div>
        <h1 style={{ fontSize: '32px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>EagleSwing</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>Unified golf swing analysis: raw iPhone footage &rarr; biomechanical insights</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px', marginBottom: '32px' }}>
        <StatCard label="Stages" value="5" sub="ingest &rarr; analyze" color={colors.accent} delay={80} />
        <StatCard label="End-to-End" value="~6 min" sub="5-min 60fps video" color={colors.purple} delay={140} />
        <StatCard label="Throughput" value="57.5 fps" sub="turbo + torch.compile" color={colors.green} delay={200} />
        <StatCard label="Cost" value="$0.04" sub="per video (spot)" color={colors.amber} delay={260} />
      </div>

      <CollapsibleCard title="Architecture" sub="How data moves from phone to analysis" icon="&#128205;">
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0' }}>
          <FlowNode type="default" title="iPhone" subtitle="Upload .MOV via HTTPS" />
          <FlowArrow label=".MOV upload" color={colors.green} />
          <FlowNode type="s3" title="S3: golf-swing-data" subtitle="{golfer}/raw/" />
          <FlowArrow label="S3 event (.MOV/.mp4)" color={colors.green} />
          <FlowNode type="sqs" title="SQS: golf-video-label" />
          <FlowArrow label="worker.py polls" color={colors.rose} />
          <FlowNode type="ec2" title="EC2 g6.2xlarge (L4 GPU)" subtitle="NVENC transcode + ViTPose-Huge" details="~57.5 fps turbo mode, torch.compile + NVDEC" />
          <div style={{ display: 'flex', gap: '24px', width: '100%', maxWidth: '420px' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <FlowArrow label=".mp4" color={colors.green} />
              <FlowNode type="s3" title="S3: /processed/" />
            </div>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <FlowArrow label=".pkl" color={colors.green} />
              <FlowNode type="s3" title="S3: /keypoints/" />
            </div>
          </div>
          <FlowArrow label="S3 event (.pkl)" color={colors.amber} />
          <FlowNode type="lambda" title="Lambda: swing_detection" subtitle="~13s signal processing" />
          <div style={{ display: 'flex', gap: '24px', width: '100%', maxWidth: '420px' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <FlowArrow label=".json" color={colors.green} />
              <FlowNode type="s3" title="S3: /detection/" />
            </div>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <FlowArrow label="PutItem" color={colors.accent} />
              <FlowNode type="db" title="DynamoDB" subtitle="golf-swing-detections" />
            </div>
          </div>
          <FlowArrow label="S3 event (.json)" color={colors.amber} />
          <FlowNode type="lambda" title="Lambda: post_processing" subtitle="fingers + frames + notifications" />
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="Modules at a Glance" icon="&#128230;">
        <Table
          headers={['Module', 'Runtime', 'What it does']}
          rows={[
            ['eagle_swing.data', 'Library', 'PKL loading (both formats), keypoint schema, session discovery'],
            ['eagle_swing.detection', 'Lambda/CLI', 'Signal processing on keypoints: backswing + contact detection'],
            ['eagle_swing.analysis', 'Lambda/CLI', 'Biomechanical metrics, SPM comparison, normalization'],
            ['eagle_swing.hand_finder', 'Lambda', 'Post-swing hand raises, finger count prediction'],
            ['eagle_swing.viz', 'CLI', 'Comparison plots, metric visualizations'],
            ['eagle_swing.pipeline', 'EC2/Lambda', 'Label worker, ingest, Lambda entry points'],
          ]}
        />
      </CollapsibleCard>

      <CollapsibleCard title="Quick Start" icon="&#9889;">
        <CodeBlock title="Install">{`git clone https://github.com/azaidi06/dev-pipe.git
cd dev-pipe
pip install -e .`}</CodeBlock>
        <CodeBlock title="Load keypoint data">{`from eagle_swing.data.loader import load_pkl

kd = load_pkl("path/to/video.pkl")
print(kd.keypoints.shape)  # (N, 17, 2)
print(kd.meta.fps)         # 60.0`}</CodeBlock>
        <CodeBlock title="Detect swings">{`from eagle_swing.detection.backswing import detect_backswings
from eagle_swing.detection.contact import detect_contacts

result = detect_backswings("video.pkl", "video.MOV")
contacts = detect_contacts(result)
print(f"{result.n_swings} swings, {contacts.n_contacts} contacts")`}</CodeBlock>
      </CollapsibleCard>

      <CollapsibleCard title="Performance Summary" icon="&#9889;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px' }}>
          <div>
            <Table
              headers={['Stage', 'Time', 'Hardware']}
              rows={[
                ['Transcode (NVENC)', '~10s', 'L4 GPU hardware encoder'],
                ['Labeling (turbo)', '~5.5 min', '57.5 fps, torch.compile + NVDEC'],
                ['Swing Detection', '~13s', 'Lambda CPU (numpy/scipy)'],
                ['End-to-end', '~6 min', '~$0.04/video on spot'],
              ]}
            />
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: colors.text, marginBottom: '12px' }}>GPU Hardware Utilization</div>
            <p style={{ fontSize: '12px', color: colors.textMuted, lineHeight: 1.6, marginBottom: '12px' }}>
              The L4 has three independent silicon blocks &mdash; using one doesn't steal from the others:
            </p>
            <Table
              headers={['Block', 'Transcode', 'Labeling']}
              rows={[
                ['NVDEC (decode)', 'Decode HEVC input', 'Decode H.264 frames'],
                ['CUDA (compute)', 'Idle', 'RTMDet + ViTPose'],
                ['NVENC (encode)', 'Encode H.264 output', 'Idle'],
              ]}
            />
          </div>
        </div>
      </CollapsibleCard>
    </div>
  );
};

export default Overview;
