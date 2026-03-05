import React from 'react';
import { colors } from '../colors';

const Badge = ({ children, color = colors.accent }) => (
  <span style={{
    fontFamily: "'JetBrains Mono', monospace", fontSize: '11px', fontWeight: 600,
    color, background: `${color}15`, padding: '3px 10px', borderRadius: '6px',
    whiteSpace: 'nowrap',
  }}>{children}</span>
);

export default Badge;
