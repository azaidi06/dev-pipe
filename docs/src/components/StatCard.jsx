import React, { useState, useEffect } from 'react';
import { colors } from '../colors';

const StatCard = ({ label, value, sub, color = colors.accent, delay = 0 }) => {
  const [visible, setVisible] = useState(false);
  useEffect(() => { const t = setTimeout(() => setVisible(true), delay); return () => clearTimeout(t); }, [delay]);

  return (
    <div style={{
      background: `linear-gradient(135deg, ${color}08, ${color}04)`,
      border: `1px solid ${color}25`,
      borderRadius: '12px', padding: '18px 20px',
      opacity: visible ? 1 : 0,
      transform: visible ? 'translateY(0)' : 'translateY(8px)',
      transition: 'all 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
    }}>
      <div style={{ fontSize: '11px', fontWeight: 600, color: colors.textDim, textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: '6px' }}>{label}</div>
      <div style={{ fontSize: '22px', fontWeight: 700, color, lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: '11px', color: colors.textDim, marginTop: '4px' }}>{sub}</div>}
    </div>
  );
};

export default StatCard;
