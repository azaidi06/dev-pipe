import React from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import Table from '../components/Table';
import CodeBlock from '../components/CodeBlock';
import useInView from '../hooks/useInView';

const PipelineStep = ({ label, sub, color, isLast }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: color, border: `2px solid ${color}40` }} />
      {!isLast && <div style={{ width: '2px', height: '32px', background: `${color}30` }} />}
    </div>
    <div style={{ paddingBottom: isLast ? 0 : '20px' }}>
      <div style={{ fontSize: '13px', fontWeight: 700, color }}>{label}</div>
      {sub && <div style={{ fontSize: '11px', color: colors.textMuted, marginTop: '2px' }}>{sub}</div>}
    </div>
  </div>
);

const SwingDetection = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });

  const pipelineSteps = [
    { label: 'Load .pkl', sub: '17 keypoints x N frames', color: colors.accent },
    { label: 'Extract wrist x,y', sub: 'Indices 9 (left), 10 (right)', color: colors.accent },
    { label: 'Interpolate', sub: 'Fill low-confidence gaps', color: colors.green },
    { label: 'Combine x + y', sub: 'Single signal per wrist', color: colors.green },
    { label: 'Savitzky-Golay smooth', sub: 'Window=9, poly=3', color: colors.purple },
    { label: 'scipy.find_peaks', sub: 'Prominence + distance filters', color: colors.purple },
    { label: 'Filter chain', sub: 'Amplitude, spacing, edge removal', color: colors.amber },
    { label: 'Backswing apex frames', sub: 'Peak frame indices', color: colors.rose },
    { label: 'Search downswing', sub: 'Velocity minimum \u2192 contact frames', color: colors.rose },
  ];

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={headerRef} style={{ marginBottom: '32px', opacity: headerInView ? 1 : 0, transform: headerInView ? 'translateY(0)' : 'translateY(16px)', transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>Swing Detection</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>Signal processing on wrist keypoints to find backswings and contacts</p>
      </div>

      <CollapsibleCard title="Signal Pipeline" sub="From raw keypoints to detected swings" icon="&#128200;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }}>
          <div>
            {pipelineSteps.map((step, i) => (
              <PipelineStep key={step.label} {...step} isLast={i === pipelineSteps.length - 1} />
            ))}
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: colors.text, marginBottom: '12px' }}>Usage</div>
            <CodeBlock>{`from eagle_swing.detection.backswing import detect_backswings
from eagle_swing.detection.contact import detect_contacts
from eagle_swing.detection.config import Config

cfg = Config()
result = detect_backswings("video.pkl", "video.MOV", config=cfg)
contacts = detect_contacts(result, config=cfg)

print(result.peak_frames)    # array([450, 1200, 2100])
print(contacts.contact_frames)  # array([520, 1270, 2170])`}</CodeBlock>
          </div>
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="Detection Data Structures" sub="DetectionResult and ContactResult" icon="&#128202;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: colors.rose, marginBottom: '8px' }}>DetectionResult</div>
            <Table
              headers={['Field', 'Type']}
              rows={[
                ['name', 'str (video name)'],
                ['peak_frames', 'ndarray (apex indices)'],
                ['smoothed', 'ndarray (full signal)'],
                ['combined', 'ndarray (pre-smooth)'],
                ['fps', 'float'],
                ['total_frames', 'int'],
                ['filter_log', 'list (trace)'],
                ['pkl_data', 'dict (keypoints)'],
              ]}
            />
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: colors.green, marginBottom: '8px' }}>ContactResult</div>
            <Table
              headers={['Field', 'Type']}
              rows={[
                ['name', 'str (video name)'],
                ['contact_frames', 'ndarray (-1 if not found)'],
                ['backswing_result', 'DetectionResult'],
                ['smoothed', 'ndarray'],
              ]}
            />
          </div>
        </div>
      </CollapsibleCard>
    </div>
  );
};

export default SwingDetection;
