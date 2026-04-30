import React, { useState, useEffect } from 'react';
import {
  ComposedChart,
  Line,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';

export default function StockChart({ symbol, apiBase }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [period, setPeriod] = useState('6mo');

  const periods = [
    { label: '1M', value: '1mo' },
    { label: '3M', value: '3mo' },
    { label: '6M', value: '6mo' },
    { label: '1Y', value: '1y' },
    { label: '2Y', value: '2y' },
    { label: '5Y', value: '5y' },
    { label: 'MAX', value: 'max' },
  ];

  useEffect(() => {
    setLoading(true);
    fetch(`${apiBase}/screener/chart/${symbol}?period=${period}`)
      .then(r => {
        if (!r.ok) throw new Error("Failed to load chart data");
        return r.json();
      })
      .then(d => {
        setData(d);
        setLoading(false);
      })
      .catch(e => {
        setError(e.message);
        setLoading(false);
      });
  }, [symbol, apiBase, period]);

  if (loading) return <div className="chart-loading" style={{ padding: '2rem', textAlign: 'center', color: '#94a3b8' }}>Loading chart for {symbol}...</div>;
  if (error) return <div className="chart-error" style={{ padding: '2rem', textAlign: 'center', color: '#ef4444' }}>Error: {error}</div>;

  return (
    <div className="stock-chart-container" style={{ marginTop: '1rem', padding: '1rem', backgroundColor: 'rgba(15, 23, 42, 0.4)', borderRadius: '8px', border: '1px solid rgba(51, 65, 85, 0.5)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.5rem' }}>
        <h4 style={{ margin: 0, color: '#f8fafc' }}>{symbol} Technical Analysis</h4>
        <div style={{ display: 'flex', gap: '4px' }}>
          {periods.map(p => (
            <button
              key={p.value}
              onClick={() => setPeriod(p.value)}
              style={{
                background: period === p.value ? '#3b82f6' : 'transparent',
                color: period === p.value ? '#fff' : '#94a3b8',
                border: '1px solid #475569',
                borderRadius: '4px',
                padding: '4px 10px',
                fontSize: '0.75rem',
                cursor: 'pointer',
                transition: 'all 0.2s'
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>
      
      {/* Price & Moving Averages */}
      <div style={{ height: 300, width: '100%', marginBottom: '1.5rem' }}>
        <ResponsiveContainer>
          <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
            <XAxis dataKey="Date" stroke="#94a3b8" tick={{ fontSize: 11 }} minTickGap={40} />
            <YAxis domain={['auto', 'auto']} stroke="#94a3b8" tick={{ fontSize: 11 }} tickFormatter={(val) => val.toLocaleString()} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '6px' }}
              itemStyle={{ color: '#e2e8f0' }}
            />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
            <Line type="monotone" dataKey="Close" stroke="#f8fafc" dot={false} strokeWidth={2} name="Close" />
            <Line type="monotone" dataKey="SMA20" stroke="#3b82f6" dot={false} strokeWidth={2} name="SMA 20" />
            <Line type="monotone" dataKey="SMA50" stroke="#f59e0b" dot={false} strokeWidth={2} name="SMA 50" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* MACD & RSI Sub-charts */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%' }}>
        <div style={{ height: 250, width: '100%', backgroundColor: 'rgba(15, 23, 42, 0.3)', borderRadius: '6px', padding: '1rem 0.5rem 0.5rem 0.5rem' }}>
          <h5 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#94a3b8', textAlign: 'center', fontWeight: 'normal' }}>MACD (12, 26, 9)</h5>
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
              <XAxis dataKey="Date" stroke="#94a3b8" tick={{ fontSize: 10 }} minTickGap={40} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 10 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '6px' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <Bar dataKey="MACD_Hist" name="Histogram">
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.MACD_Hist > 0 ? '#10b981' : '#ef4444'} />
                ))}
              </Bar>
              <Line type="monotone" dataKey="MACD" stroke="#3b82f6" dot={false} strokeWidth={1.5} name="MACD" />
              <Line type="monotone" dataKey="MACD_Signal" stroke="#f59e0b" dot={false} strokeWidth={1.5} name="Signal" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
        
        <div style={{ height: 250, width: '100%', backgroundColor: 'rgba(15, 23, 42, 0.3)', borderRadius: '6px', padding: '1rem 0.5rem 0.5rem 0.5rem' }}>
          <h5 style={{ margin: '0 0 1rem 0', fontSize: '0.9rem', color: '#94a3b8', textAlign: 'center', fontWeight: 'normal' }}>RSI (14)</h5>
          <ResponsiveContainer>
            <ComposedChart data={data} margin={{ top: 5, right: 10, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
              <XAxis dataKey="Date" stroke="#94a3b8" tick={{ fontSize: 10 }} minTickGap={40} />
              <YAxis domain={[0, 100]} stroke="#94a3b8" tick={{ fontSize: 10 }} ticks={[30, 50, 70]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '6px' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
              <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" opacity={0.5} />
              <ReferenceLine y={30} stroke="#10b981" strokeDasharray="3 3" opacity={0.5} />
              <Line type="monotone" dataKey="RSI14" stroke="#8b5cf6" dot={false} strokeWidth={2} name="RSI" />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
