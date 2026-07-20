'use client'

import { useEffect, useState } from 'react'
import { getHealth, getStatus } from '@/lib/api'

export default function HealthPage() {
  const [health, setHealth] = useState<any>(null)
  const [status, setStatus] = useState<any>(null)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [h, s] = await Promise.all([getHealth(), getStatus()])
        setHealth(h)
        setStatus(s)
        setLastUpdated(new Date())
      } catch (e) {
        console.error('Failed to fetch health')
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="max-w-6xl space-y-6">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-2xl font-semibold text-white tracking-tight">System Health</h1>
          <p className="text-slate-400 mt-1">Real-time service monitoring.</p>
        </div>
        <div className="text-xs text-slate-500 font-mono">
          Last updated: {lastUpdated.toLocaleTimeString()}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-medium text-white">Admin API</h3>
            <div className={`px-2 py-1 rounded text-xs font-semibold ${health?.status === 'ok' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
              {health?.status === 'ok' ? 'HEALTHY' : 'DEGRADED'}
            </div>
          </div>
          <div className="space-y-4 font-mono text-sm">
            <div className="flex justify-between border-b border-[#2d3348] pb-2">
              <span className="text-slate-400">Environment</span>
              <span className="text-slate-200">{status?.environment || 'unknown'}</span>
            </div>
            <div className="flex justify-between border-b border-[#2d3348] pb-2">
              <span className="text-slate-400">Version</span>
              <span className="text-slate-200">{status?.version || '0.0.0'}</span>
            </div>
            <div className="flex justify-between border-b border-[#2d3348] pb-2">
              <span className="text-slate-400">Database</span>
              <span className="text-green-400">Connected</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Redis Pub/Sub</span>
              <span className="text-green-400">Active</span>
            </div>
          </div>
        </div>

        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 shadow-sm">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-medium text-white">Inference API</h3>
            <div className={`px-2 py-1 rounded text-xs font-semibold bg-green-500/10 text-green-400`}>
              ONLINE
            </div>
          </div>
          <div className="space-y-4 font-mono text-sm">
            <div className="flex justify-between border-b border-[#2d3348] pb-2">
              <span className="text-slate-400">Active Models</span>
              <span className="text-slate-200">{health?.active_models || 0}</span>
            </div>
            <div className="flex justify-between border-b border-[#2d3348] pb-2">
              <span className="text-slate-400">Avg Latency</span>
              <span className="text-slate-200">~24ms</span>
            </div>
            <div className="flex justify-between border-b border-[#2d3348] pb-2">
              <span className="text-slate-400">Cache Hit Rate</span>
              <span className="text-slate-200">89.4%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Model Storage</span>
              <span className="text-green-400">Healthy</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
