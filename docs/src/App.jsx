import React, { useState, useRef, useEffect } from 'react';
import { colors } from './colors';
import Overview from './pages/Overview';
import LibraryAPI from './pages/LibraryAPI';
import PipelineFlow from './pages/PipelineFlow';
import LabelVideos from './pages/LabelVideos';
import SwingDetection from './pages/SwingDetection';
import Analysis from './pages/Analysis';
import DeployCosts from './pages/DeployCosts';
import DataFormats from './pages/DataFormats';
import ScoreHandComparison from './pages/ScoreHandComparison';

const TABS = [
  { id: 'overview', label: 'Overview', component: Overview },
  { id: 'api', label: 'Library API', component: LibraryAPI },
  { id: 'pipeline', label: 'Pipeline', component: PipelineFlow },
  { id: 'label', label: 'Labeling', component: LabelVideos },
  { id: 'swing', label: 'Detection', component: SwingDetection },
  { id: 'scorehand', label: 'Score-Hand', component: ScoreHandComparison },
  { id: 'analysis', label: 'Analysis', component: Analysis },
  { id: 'deploy', label: 'Deploy', component: DeployCosts },
  { id: 'data', label: 'Data', component: DataFormats },
];

export default function App() {
  const [active, setActive] = useState('overview');
  const ActiveComponent = TABS.find(t => t.id === active).component;
  const tabsRef = useRef(null);

  useEffect(() => {
    if (tabsRef.current) {
      const activeBtn = tabsRef.current.querySelector('[data-active="true"]');
      if (activeBtn) activeBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }, [active]);

  return (
    <div style={{ minHeight: '100vh', background: colors.bg }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet" />
      <style>{`
        .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
        .hide-scrollbar::-webkit-scrollbar { display: none; }
      `}</style>
      <nav style={{
        background: colors.card, borderBottom: `1px solid ${colors.cardBorder}`,
        position: 'sticky', top: 0, zIndex: 50,
      }}>
        <div style={{
          maxWidth: '1100px', margin: '0 auto', padding: '0 12px',
          display: 'flex', alignItems: 'center', height: '52px', gap: '12px',
        }}>
          <span style={{
            fontFamily: "'DM Sans', -apple-system, sans-serif", fontWeight: 700,
            fontSize: '16px', color: colors.text, letterSpacing: '-0.01em', flexShrink: 0,
          }}>
            EagleSwing
          </span>
          <div ref={tabsRef} className="hide-scrollbar" style={{
            display: 'flex', gap: '2px', overflowX: 'auto', WebkitOverflowScrolling: 'touch',
          }}>
            {TABS.map(tab => (
              <button
                key={tab.id}
                data-active={active === tab.id}
                onClick={() => setActive(tab.id)}
                style={{
                  padding: '6px 12px', borderRadius: '8px', fontSize: '12px', fontWeight: 600,
                  fontFamily: "'DM Sans', -apple-system, sans-serif", whiteSpace: 'nowrap',
                  border: active === tab.id ? `1px solid ${colors.accent}40` : '1px solid transparent',
                  background: active === tab.id ? `${colors.accent}15` : 'transparent',
                  color: active === tab.id ? colors.accent : colors.textDim,
                  cursor: 'pointer', transition: 'all 0.2s ease', outline: 'none',
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>
      <div style={{ fontFamily: "'DM Sans', 'SF Pro Display', -apple-system, sans-serif", color: colors.text }}>
        <ActiveComponent />
      </div>
    </div>
  );
}
