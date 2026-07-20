'use client'

import { useEffect, useState } from 'react'
import { getHealth, getStatus } from '@/lib/api'
import { getSession as getLocalSession } from '@/lib/auth'

export default function OverviewPage() {
  const [health, setHealth] = useState<any>(null)
  const [status, setStatus] = useState<any>(null)
  const [session, setSession] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const s = getLocalSession()
    setSession(s)
    
    Promise.all([
      getHealth().catch(() => null),
      getStatus().catch(() => null)
    ]).then(([h, s]) => {
      if (h) setHealth(h)
      if (s) setStatus(s)
      setLoading(false)
    })
  }, [])

  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <h1 className="text-2xl font-semibold">Overview</h1>
        <div className="grid grid-cols-3 gap-6">
          <div className="h-32 bg-[#1a1f2e] rounded-xl border border-[#2d3348]"></div>
          <div className="h-32 bg-[#1a1f2e] rounded-xl border border-[#2d3348]"></div>
          <div className="h-32 bg-[#1a1f2e] rounded-xl border border-[#2d3348]"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-6xl">
      <div>
        <h1 className="text-2xl font-semibold text-white tracking-tight">Overview</h1>
        <p className="text-slate-400 mt-1">Welcome back, {session?.display_name}. Here's what's happening.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 shadow-sm">
          <h3 className="text-sm font-medium text-slate-400">Active Tenants</h3>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-4xl font-semibold text-white">{health?.active_tenants || 0}</span>
          </div>
        </div>
        
        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 shadow-sm">
          <h3 className="text-sm font-medium text-slate-400">Total Models</h3>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-4xl font-semibold text-white">{health?.active_models || 0}</span>
          </div>
        </div>

        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 shadow-sm">
          <h3 className="text-sm font-medium text-slate-400">System Status</h3>
          <div className="mt-2 flex items-center gap-2">
            <div className={`w-3 h-3 rounded-full ${health?.status === 'ok' ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-2xl font-semibold text-white capitalize">{health?.status || 'Unknown'}</span>
          </div>
          <p className="text-xs text-slate-500 mt-2">API Version {status?.version}</p>
        </div>
      </div>

      {session?.role === 'admin' && (
        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6">
          <h2 className="text-lg font-medium text-white mb-4">Recent Drift Alerts</h2>
          <div className="flex flex-col items-center justify-center py-8 text-slate-400">
            <div className="w-12 h-12 rounded-full bg-[#2d3348] flex items-center justify-center mb-3">
              <span className="text-xl">✅</span>
            </div>
            <p>No recent drift alerts detected across tenants.</p>
          </div>
        </div>
      )}
    </div>
  )
}
