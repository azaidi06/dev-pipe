import React, { useState, useEffect, useRef } from 'react';
import { colors } from '../colors';

const Chevron = ({ open, color = colors.textDim }) => (
  <svg width="20" height="20" viewBox="0 0 20 20" fill="none"
    style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.35s cubic-bezier(0.16, 1, 0.3, 1)', flexShrink: 0 }}>
    <path d="M5 7.5L10 12.5L15 7.5" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const CollapsibleCard = ({ title, sub, icon, children, defaultOpen = true, badge, style: cardStyleOverride }) => {
  const [open, setOpen] = useState(defaultOpen);
  const contentRef = useRef(null);
  const [contentHeight, setContentHeight] = useState(2000);

  useEffect(() => {
    if (contentRef.current) setContentHeight(contentRef.current.scrollHeight);
  }, [children, open]);

  useEffect(() => {
    const handleResize = () => { if (contentRef.current) setContentHeight(contentRef.current.scrollHeight); };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div style={{
      background: colors.card, border: `1px solid ${colors.cardBorder}`,
      borderRadius: '12px', marginBottom: '16px', position: 'relative', overflow: 'hidden',
      transition: 'border-color 0.3s ease', ...cardStyleOverride,
    }}>
      <button onClick={() => setOpen(prev => !prev)} style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        width: '100%', padding: '16px 16px', paddingBottom: open ? '0px' : '16px',
        background: 'transparent', border: 'none', cursor: 'pointer', textAlign: 'left', outline: 'none',
        transition: 'padding-bottom 0.35s cubic-bezier(0.16, 1, 0.3, 1)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flex: 1, minWidth: 0 }}>
          {icon && <span style={{ fontSize: '18px', flexShrink: 0 }}>{icon}</span>}
          <div style={{ minWidth: 0 }}>
            <h2 style={{ fontSize: '15px', fontWeight: 700, color: colors.text, margin: 0, letterSpacing: '-0.01em' }}>{title}</h2>
            {sub && <p style={{ fontSize: '12px', color: colors.textDim, margin: '3px 0 0 0', lineHeight: 1.4, opacity: open ? 1 : 0.7, transition: 'opacity 0.3s ease' }}>{sub}</p>}
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0, marginLeft: '16px' }}>
          {badge && !open && (
            <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', fontWeight: 600, color: colors.accent, background: `${colors.accent}15`, padding: '4px 10px', borderRadius: '6px', whiteSpace: 'nowrap' }}>{badge}</span>
          )}
          <div style={{ width: '32px', height: '32px', borderRadius: '8px', background: open ? `${colors.accent}12` : `${colors.textDim}10`, display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'background 0.3s ease' }}>
            <Chevron open={open} color={open ? colors.accent : colors.textDim} />
          </div>
        </div>
      </button>
      <div style={{ maxHeight: open ? `${contentHeight + 40}px` : '0px', opacity: open ? 1 : 0, overflow: 'hidden', transition: 'max-height 0.45s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s ease' }}>
        <div ref={contentRef} style={{ padding: '14px 16px 20px 16px' }}>{children}</div>
      </div>
    </div>
  );
};

export default CollapsibleCard;
