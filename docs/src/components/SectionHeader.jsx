import React from 'react';
import { colors } from '../colors';

const SectionHeader = ({ title, sub, color = colors.accent }) => (
  <div style={{ marginBottom: '20px' }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px' }}>
      <div style={{ width: '4px', height: '24px', borderRadius: '2px', background: color }} />
      <h2 style={{ fontSize: '20px', fontWeight: 700, color: colors.text, margin: 0, letterSpacing: '-0.01em' }}>{title}</h2>
    </div>
    {sub && <p style={{ color: colors.textDim, fontSize: '13px', margin: '0 0 0 14px' }}>{sub}</p>}
  </div>
);

export default SectionHeader;
