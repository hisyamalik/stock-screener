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

  if (loading) return <div className="chart-loading" style={{ padding: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Loading P&L data...</div>;
  if (error) return <div className="chart-error" style={{ padding: '1rem', color: 'var(--danger)', fontSize: '0.9rem' }}>Error: {error}</div>;
  if (!data || data.length === 0) return <div style={{ padding: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>No performance data available yet.</div>;

  return (
    <div style={{ height: 220, width: '100%', marginTop: '1rem' }}>
      <h5 style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', fontWeight: 'normal' }}>Live P&L Curve</h5>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id={`colorEquity-${botType}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--success)" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="var(--success)" stopOpacity={0}/>
            </linearGradient>
            <linearGradient id={`colorBalance-${botType}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="var(--accent-color)" stopOpacity={0.3}/>
              <stop offset="95%" stopColor="var(--accent-color)" stopOpacity={0}/>
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--glass-border)" opacity={0.5} />
          <XAxis 
            dataKey="Timestamp" 
            stroke="var(--text-secondary)" 
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
          <YAxis domain={['auto', 'auto']} stroke="var(--text-secondary)" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: 'var(--bg-color)', borderColor: 'var(--glass-border)', color: 'var(--text-primary)', borderRadius: '6px', fontSize: '12px' }}
            itemStyle={{ color: 'var(--text-primary)' }}
            labelFormatter={(label) => `Time: ${label}`}
          />
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '5px', color: 'var(--text-secondary)' }} />
          <Area type="monotone" dataKey="Equity" stroke="var(--success)" fillOpacity={1} fill={`url(#colorEquity-${botType})`} strokeWidth={2} />
          <Area type="stepAfter" dataKey="Balance" stroke="var(--accent-color)" fillOpacity={1} fill={`url(#colorBalance-${botType})`} strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
