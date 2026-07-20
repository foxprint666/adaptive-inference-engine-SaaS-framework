'use client'

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { DriftPoint } from '../lib/api'

interface DriftChartProps {
  data: DriftPoint[]
  dataKey: 'psi' | 'adversarial_auc'
  threshold: number
  title: string
  color: string
}

const CustomDot = (props: any) => {
  const { cx, cy, payload } = props

  if (payload.retraining_triggered) {
    return (
      <svg x={cx - 8} y={cy - 12} width={16} height={16} fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" fill="#f59e0b" stroke="#f59e0b" />
      </svg>
    )
  }

  if (payload.drift_detected) {
    return (
      <circle cx={cx} cy={cy} r={4} stroke="none" fill="#ef4444" />
    )
  }

  return <circle cx={cx} cy={cy} r={3} stroke="none" fill={props.stroke} />
}

export function DriftChart({ data, dataKey, threshold, title, color }: DriftChartProps) {
  return (
    <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-5">
      <h3 className="text-lg font-medium text-slate-200 mb-4">{title}</h3>
      <div className="h-64 w-full text-xs">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2d3348" vertical={false} />
            <XAxis 
              dataKey="ts" 
              stroke="#64748b" 
              tickFormatter={(val) => val ? new Date(val).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : ''}
              minTickGap={30}
            />
            <YAxis stroke="#64748b" domain={['auto', 'auto']} tickFormatter={(v: number) => v.toFixed(2)} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1a1f2e', borderColor: '#2d3348', color: '#f1f5f9' }}
              labelFormatter={(val) => new Date(val).toLocaleString()}
              itemStyle={{ color: '#f1f5f9' }}
            />
            <ReferenceLine y={threshold} stroke="#ef4444" strokeDasharray="3 3" />
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2}
              dot={<CustomDot />}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
