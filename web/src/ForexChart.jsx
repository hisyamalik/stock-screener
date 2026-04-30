import React, { useState, useEffect } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

export default function ForexChart({ botType, apiBase }) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    fetch(`${apiBase}/forex/${botType}/performance`)
      .then(r => {
        if (!r.ok) throw new Error("Failed to load performance data");
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
  }, [botType, apiBase]);

  if (loading) return <div className="chart-loading" style={{ padding: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>Loading P&L data...</div>;
  if (error) return <div className="chart-error" style={{ padding: '1rem', color: '#ef4444', fontSize: '0.9rem' }}>Error: {error}</div>;
  if (!data || data.length === 0) return <div style={{ padding: '1rem', color: '#94a3b8', fontSize: '0.9rem' }}>No performance data available yet.</div>;

  return (
    <div style={{ height: 220, width: '100%', marginTop: '1rem' }}>
      <h5 style={{ fontSize: '0.8rem', color: '#94a3b8', marginBottom: '0.5rem', fontWeight: 'normal' }}>Live P&L Curve</h5>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={`colorEquity-${botType}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#10b981" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id={`colorBalance-${botType}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.5} />
          <XAxis 
            dataKey="Timestamp" 
            stroke="#94a3b8" 
            tick={{ fontSize: 10 }} 
            tickFormatter={(val) => {
              if(!val) return '';
              try {
                const date = new Date(val);
                if (isNaN(date.getTime())) return val.substring(0, 10);
                return `${date.getMonth() + 1}/${date.getDate()}`;
              } catch {
                return val.substring(0, 10);
              }
            }} 
          />
          <YAxis domain={['auto', 'auto']} stroke="#94a3b8" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', color: '#f8fafc', borderRadius: '6px', fontSize: '12px' }}
            itemStyle={{ color: '#e2e8f0' }}
            labelFormatter={(label) => `Time: ${label}`}
          />
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px' }} />
          <Area type="monotone" dataKey="Equity" stroke="#10b981" fillOpacity={1} fill={`url(#colorEquity-${botType})`} strokeWidth={2} />
          <Area type="stepAfter" dataKey="Balance" stroke="#3b82f6" fillOpacity={1} fill={`url(#colorBalance-${botType})`} strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
