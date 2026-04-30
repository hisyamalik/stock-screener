import React, { useState, useEffect } from 'react';
import { Activity, Play, Square, TrendingUp, Monitor, AlignEndHorizontal, LineChart, Activity as ActivityIcon } from 'lucide-react';
import StockChart from './StockChart';
import ForexChart from './ForexChart';

const API_BASE = "http://localhost:8000/api";

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  
  return (
    <div className="app-container">
      <nav className="sidebar glass-panel">
        <div className="logo-section">
          <div className="logo-icon"><TrendingUp size={24} /></div>
          <h2>QuantDash</h2>
        </div>
        <div className="nav-links">
          <button className={`nav-link ${activeTab === 'dashboard' ? 'active' : ''}`} onClick={() => setActiveTab('dashboard')}>
            <Activity size={20} /> Dashboard
          </button>
          <button className={`nav-link ${activeTab === 'forex' ? 'active' : ''}`} onClick={() => setActiveTab('forex')}>
            <Monitor size={20} /> Forex Robots
          </button>
          <button className={`nav-link ${activeTab === 'screener' ? 'active' : ''}`} onClick={() => setActiveTab('screener')}>
            <AlignEndHorizontal size={20} /> Stock Screener
          </button>
        </div>
      </nav>
      
      <main className="main-content">
        <header className="topbar glass-panel">
          <h1>{activeTab === 'dashboard' ? 'Overview' : activeTab === 'forex' ? 'Forex Trading Bots' : 'IDX Stock Screener'}</h1>
          <div className="status-badge pulse">Live Mode</div>
        </header>

        <div className="content-area">
          {activeTab === 'dashboard' && <Dashboard />}
          {activeTab === 'forex' && <ForexPanel />}
          {activeTab === 'screener' && <ScreenerPanel />}
        </div>
      </main>
    </div>
  );
}

function Dashboard() {
  return (
    <div className="dashboard-grid">
      <div className="dashboard-card glass-panel fade-in">
        <h3>System Status</h3>
        <p className="status-text healthy">All Systems Operational</p>
      </div>
      <div className="dashboard-card glass-panel fade-in delay-1">
        <h3>Active Bots</h3>
        <p className="big-number">0</p>
      </div>
      <div className="dashboard-card glass-panel fade-in delay-2">
        <h3>Latest Screener Results</h3>
        <p className="status-text warning">Run required</p>
      </div>
    </div>
  );
}

function ForexPanel() {
  const [status, setStatus] = useState({});
  const [logs, setLogs] = useState({});
  const [expandedBot, setExpandedBot] = useState(null);
  const [showLogs, setShowLogs] = useState({});

  const bots = [
    { id: 'trend', name: 'Trend Following Bot', desc: 'MT5 trend-following robot with dynamic TP/SL.' },
    { id: 'sr', name: 'Support & Resistance', desc: 'Scalping bot operating on SR zones.' },
    { id: 'po3', name: 'PO3 Strategy', desc: 'Power of 3 smart money concepts bot.' }
  ];

  const checkStatusAndLogs = async () => {
    try {
      const res = await fetch(`${API_BASE}/forex/status`);
      const statusData = await res.json();
      setStatus(statusData);
      
      setExpandedBot(prev => {
        if (prev) return prev;
        const runningBot = Object.keys(statusData).find(k => statusData[k] === 'running');
        return runningBot || null;
      });

      const logsData = {};
      for (const bot of bots) {
        try {
          const logRes = await fetch(`${API_BASE}/forex/${bot.id}/logs`);
          const logJson = await logRes.json();
          logsData[bot.id] = logJson.logs;
        } catch(e) {}
      }
      setLogs(logsData);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    checkStatusAndLogs();
    const interval = setInterval(checkStatusAndLogs, 2000);
    return () => clearInterval(interval);
  }, []);

  const handleAction = async (bot, action) => {
    if (action === 'start') {
        setStatus(prev => ({...prev, [bot]: 'starting'}));
    }
    await fetch(`${API_BASE}/forex/${bot}/${action}`, { method: 'POST' });
    checkStatusAndLogs();
  };

  return (
    <div className="bot-grid" style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {bots.map((bot, i) => {
        const isExpanded = expandedBot === bot.id;
        return (
          <div key={bot.id} className={`bot-card glass-panel fade-in target-delay-${i}`} style={{ overflow: 'hidden' }}>
            <div className="bot-header" 
                 style={{ cursor: 'pointer', userSelect: 'none', paddingBottom: isExpanded ? '1rem' : '0' }} 
                 onClick={() => setExpandedBot(isExpanded ? null : bot.id)}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '1.2rem', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s', display: 'inline-block' }}>▶</span>
                <h3 style={{ margin: 0 }}>{bot.name}</h3>
              </div>
              <span className={`status-indicator ${status[bot.id] === 'running' ? 'running' : 'stopped'}`}>
                {status[bot.id] === 'running' ? 'Active' : status[bot.id] === 'starting' ? 'Starting...' : 'Offline'}
              </span>
            </div>
            
            {isExpanded && (
              <div className="bot-content fade-in" style={{ borderTop: '1px solid rgba(51, 65, 85, 0.5)', paddingTop: '1rem' }}>
                <p>{bot.desc}</p>
                <div className="bot-actions" style={{ display: 'flex', gap: '1rem', alignItems: 'center', flexWrap: 'wrap' }}>
                  {status[bot.id] !== 'running' && status[bot.id] !== 'starting' ? (
                    <button className="btn btn-primary" onClick={() => handleAction(bot.id, 'start')}>
                      <Play size={16} /> Start
                    </button>
                  ) : (
                    <button className="btn btn-danger" onClick={() => handleAction(bot.id, 'stop')}>
                      <Square size={16} /> Stop
                    </button>
                  )}
                  <button className="btn" style={{ background: 'transparent', border: '1px solid #475569', color: '#f8fafc' }} 
                          onClick={() => setShowLogs(prev => ({...prev, [bot.id]: !prev[bot.id]}))}>
                    <ActivityIcon size={16} /> {showLogs[bot.id] ? 'Hide Log' : 'Show Log'}
                  </button>
                </div>
                
                {/* Performance Chart */}
                <div style={{ marginTop: '1.5rem', borderTop: '1px solid rgba(51, 65, 85, 0.2)', paddingTop: '1rem' }}>
                  <ForexChart botType={bot.id} apiBase={API_BASE} />
                </div>

                {showLogs[bot.id] && logs[bot.id] && (
                  <div className="bot-console fade-in" style={{ marginTop: '1rem' }}>
                    <pre>{logs[bot.id]}</pre>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ScreenerPanel() {
  const [data, setData] = useState([]);
  const [status, setStatus] = useState('idle');
  const [logs, setLogs] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [showLogs, setShowLogs] = useState(false);

  const checkStatus = () => {
    fetch(`${API_BASE}/screener/status`)
      .then(r => r.json())
      .then(d => setStatus(d.status))
      .catch(console.error);
      
    fetch(`${API_BASE}/screener/logs`)
      .then(r => r.json())
      .then(d => setLogs(d.logs))
      .catch(console.error);
  };

  const loadReport = () => {
    fetch(`${API_BASE}/screener/report`)
      .then(r => {
        if (!r.ok) throw new Error("Not found");
        return r.json();
      })
      .then(d => setData(d))
      .catch(e => console.log(e));
  }

  useEffect(() => {
    checkStatus();
    loadReport();
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const runScreener = async () => {
    setStatus('running');
    setLogs('Starting screener...\n');
    await fetch(`${API_BASE}/screener/run`, { method: 'POST' });
  };

  return (
    <div className="screener-container fade-in">
      <div className="glass-panel screener-controls">
        <div className="screener-info">
          <h3>IDX Screener</h3>
          <p>Scans the Indonesian stock exchange for trends & momentum breakouts.</p>
        </div>
        <button 
          className="btn btn-primary" 
          onClick={runScreener}
          disabled={status === 'running'}
        >
          {status === 'running' ? 'Scanning...' : 'Run Screener'}
        </button>
      </div>

      {(status === 'running' || logs) && (
        <div className="glass-panel console-container fade-in delay-1">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: showLogs ? '0.5rem' : '0' }}>
            <h4 style={{ margin: 0 }}>Execution Logs</h4>
            <button className="btn" style={{ padding: '4px 8px', fontSize: '0.8rem', background: 'transparent', border: '1px solid #475569', color: '#f8fafc' }} onClick={() => setShowLogs(!showLogs)}>
              {showLogs ? 'Hide Log' : 'Show Log'}
            </button>
          </div>
          {showLogs && <pre className="console-log fade-in" style={{ marginTop: '0.5rem' }}>{logs}</pre>}
        </div>
      )}

      {data.length > 0 && (
        <div className="glass-panel table-container fade-in delay-1">
          <table className="data-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Action</th>
                <th>Score</th>
                <th>Close</th>
                <th>Chg %</th>
                <th>Vol Ratio</th>
                <th>Notes</th>
              </tr>
            </thead>
            <tbody>
              {data.slice(0, 15).map((row, i) => (
                <React.Fragment key={i}>
                  <tr className="table-row">
                    <td className="symbol-cell" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <button 
                        className="btn btn-icon" 
                        style={{ padding: '4px', background: 'transparent', border: 'none', color: '#94a3b8', cursor: 'pointer' }}
                        title="Toggle Chart"
                        onClick={() => setSelectedSymbol(selectedSymbol === row.symbol ? null : row.symbol)}
                      >
                        <LineChart size={16} color={selectedSymbol === row.symbol ? '#3b82f6' : 'currentColor'} />
                      </button>
                      {row.symbol}
                    </td>
                    <td>
                      <span className={`badge ${row.action.toLowerCase()}`}>{row.action}</span>
                    </td>
                    <td>{row.total_score}</td>
                    <td>{row.close.toLocaleString()}</td>
                    <td className={row.change_pct > 0 ? 'positive' : row.change_pct < 0 ? 'negative' : ''}>
                      {row.change_pct > 0 ? '+' : ''}{row.change_pct.toFixed(2)}%
                    </td>
                    <td>{row.vol_ratio.toFixed(2)}x</td>
                    <td className="notes-cell">{row.notes}</td>
                  </tr>
                  {selectedSymbol === row.symbol && (
                    <tr className="chart-row fade-in">
                      <td colSpan="7" style={{ padding: 0 }}>
                        <StockChart symbol={row.symbol} apiBase={API_BASE} />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default App;
