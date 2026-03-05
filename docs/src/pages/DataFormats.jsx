import React from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import Table from '../components/Table';
import CodeBlock from '../components/CodeBlock';
import useInView from '../hooks/useInView';

const DataFormats = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });

  const cocoMap = [
    [0, 'Nose'], [1, 'Left Eye'], [2, 'Right Eye'], [3, 'Left Ear'], [4, 'Right Ear'],
    [5, 'Left Shoulder'], [6, 'Right Shoulder'], [7, 'Left Elbow'], [8, 'Right Elbow'],
    [9, 'Left Wrist'], [10, 'Right Wrist'], [11, 'Left Hip'], [12, 'Right Hip'],
    [13, 'Left Knee'], [14, 'Right Knee'], [15, 'Left Ankle'], [16, 'Right Ankle'],
  ];

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={headerRef} style={{ marginBottom: '32px', opacity: headerInView ? 1 : 0, transform: headerInView ? 'translateY(0)' : 'translateY(16px)', transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>Data Formats</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>PKL structure, JSON schemas, and keypoint conventions</p>
      </div>

      <CollapsibleCard title="PKL Keypoint Files" sub="Core data artifact &mdash; one per video" icon="&#128230;">
        <CodeBlock title="Pipeline format (frame_N keys + __meta__)">{`{
    "frame_0": {
        "keypoints": np.ndarray(17, 2),       # COCO-17 (x, y) per frame
        "keypoint_scores": np.ndarray(17,),    # Confidence [0, 1] per joint
    },
    "frame_1": {...},
    ...
    "__meta__": {
        "fps": 60.0,
        "total_frames": 3000,
        "width": 1080,
        "height": 1920,
        "n_pkl_frames": 3000
    }
}`}</CodeBlock>
        <CodeBlock title="Eagle-swing format (list/dict without __meta__)">{`[
    {"keypoints": np.ndarray(17, 2), "keypoint_scores": np.ndarray(17,)},
    {"keypoints": np.ndarray(17, 2), "keypoint_scores": np.ndarray(17,)},
    ...
]`}</CodeBlock>
        <p style={{ fontSize: '12px', color: colors.textDim, lineHeight: 1.6 }}>
          <code style={{ color: colors.accent, background: `${colors.accent}15`, padding: '1px 6px', borderRadius: '4px', fontSize: '11px' }}>eagle_swing.data.loader.load_pkl()</code> handles both formats transparently, returning a unified <code style={{ color: colors.accent, background: `${colors.accent}15`, padding: '1px 6px', borderRadius: '4px', fontSize: '11px' }}>KeypointData</code> object.
        </p>
      </CollapsibleCard>

      <CollapsibleCard title="COCO-17 Keypoint Map" icon="&#129524;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '16px' }}>
          <div>
            {cocoMap.slice(0, 9).map(([idx, name]) => (
              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '6px 0', borderBottom: `1px solid ${colors.divider}` }}>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', fontWeight: 600, color: colors.accent, minWidth: '24px', textAlign: 'right' }}>{idx}</span>
                <span style={{ fontSize: '13px', color: colors.textMuted }}>{name}</span>
              </div>
            ))}
          </div>
          <div>
            {cocoMap.slice(9).map(([idx, name]) => {
              const isWrist = idx === 9 || idx === 10;
              return (
                <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '10px', padding: '6px 0', borderBottom: `1px solid ${colors.divider}` }}>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', fontWeight: 600, color: isWrist ? colors.rose : colors.accent, minWidth: '24px', textAlign: 'right' }}>{idx}</span>
                  <span style={{ fontSize: '13px', color: isWrist ? colors.text : colors.textMuted, fontWeight: isWrist ? 600 : 400 }}>{name}</span>
                  {isWrist && <span style={{ fontSize: '10px', color: colors.rose, background: `${colors.rose}15`, padding: '2px 8px', borderRadius: '4px' }}>swing signal</span>}
                </div>
              );
            })}
          </div>
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="KeypointData Schema" sub="Unified data object from load_pkl()" icon="&#128196;">
        <CodeBlock>{`from eagle_swing.data.schema import KeypointData, SwingMeta

# KeypointData fields:
#   keypoints: np.ndarray  (N, 17, 2)  - x, y pixel coords
#   scores:    np.ndarray  (N, 17)     - confidence per joint
#   meta:      SwingMeta               - metadata

# SwingMeta fields:
#   pkl_path:     str
#   fps:          float | None
#   total_frames: int | None
#   n_pkl_frames: int
#   video_path:   str
#   extra:        dict        - raw __meta__ contents

# Convenience methods:
kd = load_pkl("video.pkl")
len(kd)           # number of frames
kd.frame(0)       # (17, 3) array: [x, y, score]
kd.slice(10, 20)  # KeypointData for frames 10-19
kd.raw            # (N, 17, 3) array with scores`}</CodeBlock>
      </CollapsibleCard>
    </div>
  );
};

export default DataFormats;
