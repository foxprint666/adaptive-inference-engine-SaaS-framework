'use client'

import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { getDriftHistory, triggerRetrain, getJobStatus, DriftHistoryResponse, DriftPoint } from '@/lib/api'
import { getSession } from '@/lib/auth'
import { useToast } from '@/components/toast'
import { DriftChart } from '@/components/DriftChart'

export default function DriftPage() {
  const params = useParams()
  const tenantId = params.tenantId as string
  const modelId = params.modelId as string
  
  const [data, setData] = useState<DriftHistoryResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [retrainJob, setRetrainJob] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<string | null>(null)
  const { addToast } = useToast()

  const loadData = async () => {
    const session = getSession()
    if (!session) return
    try {
      const res = await getDriftHistory(session.token, tenantId, modelId)
      setData(res)
    } catch (err: unknown) {
      const msg = (err as { message?: string }).message || 'Failed to load drift history'
      addToast(msg, 'error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const interval = setInterval(loadData, 30000) // Poll every 30s
    return () => clearInterval(interval)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tenantId, modelId])

  useEffect(() => {
    if (!retrainJob) return
    
    const checkJob = async () => {
      const session = getSession()
      if (!session) return
      try {
        const res = await getJobStatus(session.token, tenantId, retrainJob)
        setJobStatus(res.status)
        if (res.status === 'completed' || res.status === 'failed') {
          addToast(`Retrain job ${res.status}`, res.status === 'completed' ? 'success' : 'error')
          setRetrainJob(null)
          loadData()
        }
      } catch (err: unknown) {
        const msg = (err as { message?: string }).message || 'Job status check failed'
        addToast(msg, 'error')
      }
    }

    const interval = setInterval(checkJob, 3000)
    return () => clearInterval(interval)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [retrainJob, tenantId])

  const handleRetrain = async () => {
    const session = getSession()
    if (!session) return
    try {
      const res = await triggerRetrain(session.token, tenantId, modelId, 'Manual retrain trigger')
      setRetrainJob(res.job_id)
      setJobStatus('queued')
      addToast(`Retraining job started: ${res.job_id}`, 'info')
    } catch (err: unknown) {
      const msg = (err as { message?: string }).message || 'Failed to trigger retrain'
      addToast(msg, 'error')
    }
  }

  // Derive summary stats from history array
  const lastPoint: DriftPoint | null = data?.history?.length ? data.history[data.history.length - 1] : null
  const currentPsi = lastPoint?.psi ?? 0
  const currentAuc = lastPoint?.adversarial_auc ?? 0.5
  const isDrifting = lastPoint?.drift_detected ?? false
  const lastCheck = lastPoint?.ts ? new Date(lastPoint.ts).toLocaleString() : 'N/A'

  if (loading) return (
    <div className="p-8 space-y-4">
      <div className="h-8 w-64 bg-[#1a1f2e] rounded animate-pulse" />
      <div className="grid grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="h-28 bg-[#1a1f2e] rounded-xl animate-pulse" />
        ))}
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="h-72 bg-[#1a1f2e] rounded-xl animate-pulse" />
        <div className="h-72 bg-[#1a1f2e] rounded-xl animate-pulse" />
      </div>
    </div>
  )

  if (!data) return (
    <div className="p-8 text-center">
      <div className="text-red-400 text-lg font-medium">Failed to load drift data</div>
      <p className="text-slate-500 mt-2 text-sm">Ensure DATABASE_URL is set and the drift worker has run at least one check cycle.</p>
      <button onClick={loadData} className="mt-4 px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm hover:bg-indigo-700">
        Retry
      </button>
    </div>
  )

  return (
    <div className="max-w-7xl space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-2xl font-semibold text-white tracking-tight">Drift Monitor</h1>
          <p className="text-slate-400 mt-1 font-mono text-sm">Tenant: {tenantId} | Model: {modelId}</p>
        </div>
        <button
          onClick={handleRetrain}
          disabled={!!retrainJob}
          className="bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed text-white px-5 py-2.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
        >
          {retrainJob ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Retraining ({jobStatus})
            </>
          ) : (
            <>⚡ Force Retrain</>
          )}
        </button>
      </div>

      {/* KPI summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6">
          <h3 className="text-sm font-medium text-slate-400">Current PSI</h3>
          <div className="mt-2">
            <span className={`text-3xl font-semibold ${currentPsi > 0.1 ? 'text-red-400' : 'text-emerald-400'}`}>
              {currentPsi.toFixed(4)}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">{currentPsi > 0.25 ? '🔴 Critical' : currentPsi > 0.1 ? '🟡 Warning' : '🟢 Normal'}</p>
        </div>
        
        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6">
          <h3 className="text-sm font-medium text-slate-400">Adversarial AUC</h3>
          <div className="mt-2">
            <span className={`text-3xl font-semibold ${currentAuc > 0.7 ? 'text-red-400' : 'text-emerald-400'}`}>
              {currentAuc.toFixed(4)}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">{currentAuc > 0.7 ? '🔴 Drift likely' : '🟢 Stable'}</p>
        </div>

        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6">
          <h3 className="text-sm font-medium text-slate-400">Drift Status</h3>
          <div className="mt-2 flex items-center gap-2">
            {isDrifting ? (
              <>
                <span className="w-3 h-3 rounded-full bg-red-500 animate-pulse" />
                <span className="text-xl font-semibold text-red-400">Detected</span>
              </>
            ) : (
              <>
                <span className="w-3 h-3 rounded-full bg-emerald-500" />
                <span className="text-xl font-semibold text-emerald-400">Healthy</span>
              </>
            )}
          </div>
        </div>

        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6">
          <h3 className="text-sm font-medium text-slate-400">Data Points</h3>
          <div className="mt-2">
            <span className="text-3xl font-semibold text-white">{data.points}</span>
          </div>
          <p className="text-xs text-slate-500 mt-1">Last: {lastCheck}</p>
        </div>
      </div>

      {/* Drift charts */}
      {data.history.length === 0 ? (
        <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-12 text-center">
          <div className="text-4xl mb-3">📊</div>
          <p className="text-slate-400 font-medium">No drift history yet</p>
          <p className="text-slate-500 text-sm mt-1">The drift worker needs to run at least one check cycle to populate charts.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <DriftChart
            data={data.history}
            dataKey="psi"
            threshold={0.25}
            title="Population Stability Index (PSI)"
            color="#6366f1"
          />
          <DriftChart
            data={data.history}
            dataKey="adversarial_auc"
            threshold={0.72}
            title="Adversarial AUC"
            color="#a855f7"
          />
        </div>
      )}

      {/* Drift reasons table */}
      {lastPoint?.drift_reasons && lastPoint.drift_reasons.length > 0 && (
        <div className="bg-[#1a1f2e] border border-red-900/40 rounded-xl p-5">
          <h3 className="text-sm font-semibold text-red-400 mb-3">⚠️ Active Drift Reasons</h3>
          <ul className="space-y-1">
            {lastPoint.drift_reasons.map((reason, i) => (
              <li key={i} className="text-sm text-slate-300 font-mono bg-red-950/30 rounded px-3 py-1.5">{reason}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
