import React from 'react';
import { colors } from '../colors';

const CodeBlock = ({ children, title }) => (
  <div style={{ marginBottom: '16px' }}>
    {title && <div style={{ fontSize: '11px', fontWeight: 600, color: colors.textDim, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: '6px' }}>{title}</div>}
    <pre style={{
      background: colors.card, border: `1px solid ${colors.cardBorder}`,
      borderRadius: '12px', padding: '16px 20px', margin: 0, overflow: 'auto',
      fontFamily: "'JetBrains Mono', monospace", fontSize: '12px', lineHeight: 1.6,
      color: colors.text,
    }}>
      <code>{children}</code>
    </pre>
  </div>
);

export default CodeBlock;
