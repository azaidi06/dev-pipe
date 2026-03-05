import React from 'react';
import { colors } from '../colors';
import CollapsibleCard from '../components/CollapsibleCard';
import Table from '../components/Table';
import CodeBlock from '../components/CodeBlock';
import useInView from '../hooks/useInView';

const LibraryAPI = () => {
  const [headerRef, headerInView] = useInView({ threshold: 0.1 });

  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', padding: '24px 16px' }}>
      <div ref={headerRef} style={{ marginBottom: '32px', opacity: headerInView ? 1 : 0, transform: headerInView ? 'translateY(0)' : 'translateY(16px)', transition: 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1)' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 700, margin: '0 0 6px', background: `linear-gradient(135deg, ${colors.text}, ${colors.textMuted})`, WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', letterSpacing: '-0.02em' }}>Library API</h1>
        <p style={{ color: colors.textDim, fontSize: '14px', margin: 0 }}>eagle_swing package reference &mdash; pip install -e .</p>
      </div>

      <CollapsibleCard title="eagle_swing.data" sub="PKL loading, keypoint schema, session discovery" icon="&#128230;">
        <Table
          headers={['Function / Class', 'Description']}
          rows={[
            ['load_pkl(path)', 'Load pkl file (both formats) \u2192 KeypointData'],
            ['load_pkl_raw(path)', 'Load raw pkl dict for pipeline functions'],
            ['interpolate_and_smooth(kd)', 'Savgol-smooth keypoints, interpolate gaps'],
            ['discover_sessions(root)', 'Find session directories under data root'],
            ['KeypointData', 'Unified container: .keypoints (N,17,2), .scores (N,17), .meta'],
            ['SwingMeta', 'Metadata: pkl_path, fps, total_frames, video_path'],
          ]}
        />
        <CodeBlock>{`from eagle_swing.data.loader import load_pkl, interpolate_and_smooth
from eagle_swing.data.schema import KeypointData

kd = load_pkl("video.pkl")
len(kd)              # frame count
kd.frame(0)          # (17, 3) with scores
kd.slice(10, 20)     # sub-range KeypointData
kd.raw               # (N, 17, 3)
smoothed = interpolate_and_smooth(kd)`}</CodeBlock>
      </CollapsibleCard>

      <CollapsibleCard title="eagle_swing.detection" sub="Backswing apex + contact/impact detection" icon="&#127919;">
        <Table
          headers={['Function / Class', 'Description']}
          rows={[
            ['detect_backswings(pkl, mov)', 'Detect backswing apex frames \u2192 DetectionResult'],
            ['detect_contacts(result)', 'Detect impact frames following backswings \u2192 ContactResult'],
            ['Config()', 'Frozen dataclass with ~40 tunable parameters'],
            ['DetectionResult', 'peak_frames, smoothed, combined, fps, filter_log, ...'],
            ['ContactResult', 'contact_frames, backswing_result, smoothed, filter_log'],
          ]}
        />
        <CodeBlock>{`from eagle_swing.detection.backswing import detect_backswings
from eagle_swing.detection.contact import detect_contacts

result = detect_backswings("video.pkl", "video.MOV")
print(f"{result.n_swings} swings at frames {result.peak_frames}")

contacts = detect_contacts(result)
print(f"{contacts.n_contacts} contacts")`}</CodeBlock>
      </CollapsibleCard>

      <CollapsibleCard title="eagle_swing.detection.events" sub="Address frame detection (eagle-swing methods)" icon="&#128205;">
        <Table
          headers={['Function', 'Description']}
          rows={[
            ['find_address_velocity(kd, bs)', 'Velocity-based: wrist speed minimum before backswing'],
            ['find_address_robust(kd, bs)', 'Scale-invariant: stillness scoring over multiple keypoints'],
            ['find_address_optimized(kd, bs)', 'Combined geometric + velocity with normalization'],
          ]}
        />
      </CollapsibleCard>

      <CollapsibleCard title="eagle_swing.analysis" sub="Biomechanical metrics, SPM, normalization" icon="&#128202;">
        <Table
          headers={['Function / Class', 'Description']}
          rows={[
            ['compute_swing_metrics(kps)', '17 biomechanical metrics from (N,17,2) keypoints'],
            ['perform_spm_1d(a, b)', '1D SPM t-test between two groups of time series'],
            ['perform_hotellings_t2(a, b)', '2D Hotelling T\u00b2 test for keypoint trajectories'],
            ['SwingData(row)', 'Single swing segment with computed metrics'],
            ['load_golfer_data(csv, ...)', 'Load golfer CSV + build DataFrames'],
            ['build_swing_items(df)', 'DataFrame \u2192 list of SwingData objects'],
            ['perform_group_analysis(...)', 'Full group comparison with SPM'],
          ]}
        />
        <div style={{ fontSize: '13px', fontWeight: 700, color: colors.text, marginTop: '16px', marginBottom: '8px' }}>Sub-modules</div>
        <Table
          headers={['Module', 'Key Functions']}
          rows={[
            ['analysis.normalization', 'torso_diagonal, average_torso, robust_median, procrustes'],
            ['analysis.upper_body', 'compute_upper_metrics() \u2192 rotation, tilt, X-factor'],
            ['analysis.lower_body', 'compute_lower_metrics() \u2192 knee angles, weight shift'],
            ['analysis.kinematics', 'find_crossing_frames(), find_top_of_backswing()'],
            ['analysis.temporal', 'add_derivatives(), compute_metric_derivatives()'],
          ]}
        />
      </CollapsibleCard>

      <CollapsibleCard title="eagle_swing.hand_finder" sub="Post-swing hand raise detection + finger counting" icon="&#9996;">
        <Table
          headers={['Function / Class', 'Description']}
          rows={[
            ['find_score_hands(kps, scores, cfg)', 'Detect hand raises in post-swing region'],
            ['find_plateau(wrist_y)', 'Find stable raised-hand plateau'],
            ['load_finger_model(path)', 'Load EfficientNet-B0 (ONNX or PyTorch)'],
            ['predict_finger_count(model, frame)', 'Predict fingers shown \u2192 int'],
            ['HandFinderConfig()', 'Frozen config: thresholds, keypoint indices'],
            ['HandFinderResult', 'hand_frame_range, side, representative_frame'],
          ]}
        />
      </CollapsibleCard>

      <CollapsibleCard title="eagle_swing.viz" sub="Plotting utilities" icon="&#127912;">
        <Table
          headers={['Function', 'Description']}
          rows={[
            ['plot_upper_body_comparison(src, tgt)', 'Compare upper body metrics between groups'],
            ['plot_lower_body_comparison(src, tgt)', 'Compare lower body metrics between groups'],
            ['plot_swing_metrics(metrics, title)', 'Plot all metrics for a single swing'],
          ]}
        />
      </CollapsibleCard>

      <CollapsibleCard title="eagle_swing.utils" sub="Shared constants and geometry" icon="&#128295;">
        <Table
          headers={['Name', 'Description']}
          rows={[
            ['COCO_KP', 'Dict mapping keypoint index \u2192 name (0: "nose", ...)'],
            ['COCO_BODY', 'Dict mapping body part name \u2192 index (5-16 only)'],
            ['angle_2pts(a, b)', 'Signed angle (degrees) of vector a\u2192b vs horizontal'],
            ['angle_3pts(a, b, c)', 'Angle at vertex b formed by a-b-c'],
          ]}
        />
      </CollapsibleCard>

      <CollapsibleCard title="Scripts" sub="CLI entry points in scripts/" icon="&#128187;">
        <CodeBlock title="Batch swing detection">{`python scripts/run_detection.py /path/to/dataset --contact --csv`}</CodeBlock>
        <CodeBlock title="Analysis pipeline">{`python scripts/run_analysis.py --golfer ymirza --day oct25 --phases backswing downswing`}</CodeBlock>
        <CodeBlock title="Local video labeling">{`python scripts/run_local_label.py /path/to/videos --workers 3 --skip-existing`}</CodeBlock>
      </CollapsibleCard>
    </div>
  );
};

export default LibraryAPI;
