import React from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import Table from '../components/Table';
import CodeBlock from '../components/CodeBlock';
import useInView from '../hooks/useInView';

const SubStep = ({ label, desc, color = colors.purple }) => (
  <div style={{ background: `${color}08`, border: `1px solid ${color}20`, borderRadius: '8px', padding: '10px 14px' }}>
    <div style={{ fontSize: '12px', fontWeight: 700, color }}>{label}</div>
    <div style={{ fontSize: '11px', color: colors.textMuted, marginTop: '2px', lineHeight: 1.5 }}>{desc}</div>
  </div>
);

const KeypointDot = ({ name, idx, x, y, isWrist }) => (
  <g>
    <circle cx={x} cy={y} r={isWrist ? 6 : 4} fill={isWrist ? colors.rose : colors.accent} opacity={isWrist ? 1 : 0.7} />
    <text x={x + 10} y={y + 4} fill={colors.textMuted} fontSize="9" fontFamily="'JetBrains Mono', monospace">{idx}: {name}</text>
  </g>
);

const LabelVideos = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });

  const keypoints = [
    { name: 'nose', x: 150, y: 20 }, { name: 'L eye', x: 140, y: 12 }, { name: 'R eye', x: 160, y: 12 },
    { name: 'L ear', x: 128, y: 20 }, { name: 'R ear', x: 172, y: 20 },
    { name: 'L shoulder', x: 115, y: 60 }, { name: 'R shoulder', x: 185, y: 60 },
    { name: 'L elbow', x: 95, y: 100 }, { name: 'R elbow', x: 205, y: 100 },
    { name: 'L wrist', x: 80, y: 140 }, { name: 'R wrist', x: 220, y: 140 },
    { name: 'L hip', x: 125, y: 140 }, { name: 'R hip', x: 175, y: 140 },
    { name: 'L knee', x: 120, y: 195 }, { name: 'R knee', x: 180, y: 195 },
    { name: 'L ankle', x: 115, y: 245 }, { name: 'R ankle', x: 185, y: 245 },
  ];

  const skeleton = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={headerRef} style={{ marginBottom: '32px', opacity: headerInView ? 1 : 0, transform: headerInView ? 'translateY(0)' : 'translateY(16px)', transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>Label Videos</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>NVENC transcode + ViTPose-Huge pose estimation on EC2 GPU</p>
      </div>

      <CollapsibleCard title="Pose Estimation" sub="ViTPose-Huge (632M params) + RTMDet-M detector" icon="&#129524;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '24px' }}>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: colors.text, marginBottom: '8px' }}>COCO-17 Keypoint Layout</div>
            <svg viewBox="0 0 300 265" style={{ width: '100%', maxWidth: '300px', background: `${colors.card}`, border: `1px solid ${colors.cardBorder}`, borderRadius: '12px', padding: '8px' }}>
              {skeleton.map(([i, j], idx) => (
                <line key={idx} x1={keypoints[i].x} y1={keypoints[i].y} x2={keypoints[j].x} y2={keypoints[j].y} stroke={colors.textDim} strokeWidth="1.5" opacity="0.4" />
              ))}
              {keypoints.map((kp, idx) => (
                <KeypointDot key={idx} idx={idx} name={kp.name} x={kp.x} y={kp.y} isWrist={idx === 9 || idx === 10} />
              ))}
            </svg>
            <p style={{ fontSize: '11px', color: colors.textDim, marginTop: '8px' }}>
              <span style={{ color: colors.rose, fontWeight: 700 }}>Red</span> = wrists (indices 9, 10) &mdash; primary signal for swing detection
            </p>
          </div>
          <div>
            <div style={{ fontSize: '13px', fontWeight: 700, color: colors.text, marginBottom: '8px' }}>Output per frame</div>
            <Table
              headers={['Field', 'Shape', 'Description']}
              rows={[
                ['keypoints', '(17, 2)', 'COCO-17 (x, y) pixel coords'],
                ['keypoint_scores', '(17,)', 'Confidence [0, 1] per joint'],
                ['bbox', '[x1,y1,x2,y2]', 'Person bounding box'],
              ]}
            />
          </div>
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="Turbo Mode" sub="Fully batched with threaded CPU-GPU overlap" icon="&#128640;">
        <div style={{ display: 'flex', gap: '16px', marginBottom: '16px' }}>
          <div style={{ flex: 1, padding: '16px', borderRadius: '12px', background: `${colors.accent}06`, border: `1px solid ${colors.accent}15` }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: colors.textDim, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>Background Thread (CPU)</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <SubStep label="NVDEC decode" desc="h264_cuvid hardware decode" color={colors.accent} />
              <SubStep label="Resize/pad" desc="640x640 for RTMDet" color={colors.accent} />
              <SubStep label="Normalize" desc="Batch into queue (maxsize=2)" color={colors.accent} />
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <svg width="32" height="24"><polygon points="32,12 20,0 20,24" fill={colors.accent} opacity="0.4" /></svg>
          </div>
          <div style={{ flex: 1, padding: '16px', borderRadius: '12px', background: `${colors.purple}06`, border: `1px solid ${colors.purple}15` }}>
            <div style={{ fontSize: '11px', fontWeight: 600, color: colors.textDim, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '8px' }}>Main Thread (GPU)</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              <SubStep label="RTMDet batched" desc="Inductor compiled detection" color={colors.purple} />
              <SubStep label="Crop + warp" desc="192x256 patches" color={colors.purple} />
              <SubStep label="ViTPose batched" desc="Inductor compiled pose" color={colors.purple} />
            </div>
          </div>
        </div>
      </CollapsibleCard>

      <CollapsibleCard title="torch.compile Performance" sub="Inductor compilation on L4 GPU" icon="&#9889;">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '16px' }}>
          {[
            { model: 'RTMDet', eager: '236ms', compiled: '107ms', speedup: '2.2x', color: colors.amber },
            { model: 'ViTPose', eager: '258ms', compiled: '237ms', speedup: '1.08x', color: colors.purple },
            { model: 'E2E turbo', eager: '32.5 fps', compiled: '57.5 fps', speedup: '1.77x', color: colors.green },
          ].map(item => (
            <div key={item.model} style={{ padding: '16px', borderRadius: '12px', background: `${item.color}06`, border: `1px solid ${item.color}15`, textAlign: 'center' }}>
              <div style={{ fontSize: '13px', fontWeight: 700, color: item.color, marginBottom: '10px' }}>{item.model}</div>
              <div style={{ display: 'flex', justifyContent: 'space-around', marginBottom: '8px' }}>
                <div>
                  <div style={{ fontSize: '10px', color: colors.textDim, textTransform: 'uppercase' }}>Eager</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '14px', fontWeight: 600, color: colors.textMuted }}>{item.eager}</div>
                </div>
                <div>
                  <div style={{ fontSize: '10px', color: colors.textDim, textTransform: 'uppercase' }}>Compiled</div>
                  <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '14px', fontWeight: 600, color: colors.text }}>{item.compiled}</div>
                </div>
              </div>
              <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '20px', fontWeight: 700, color: item.color }}>{item.speedup}</div>
            </div>
          ))}
        </div>
      </CollapsibleCard>
    </div>
  );
};

export default LabelVideos;
