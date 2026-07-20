'use client'

import { useState, useEffect } from 'react'
import { predict, listTenants, listTenantModels, Tenant, Model } from '@/lib/api'
import { getSession } from '@/lib/auth'
import { useToast } from '@/components/toast'

interface PredictionRecord {
  id: string
  features: string
  prediction: number
  probability: number
  latency: number
  timestamp: string
}

export default function PlaygroundPage() {
  const [tenants, setTenants] = useState<Tenant[]>([])
  const [models, setModels] = useState<Model[]>([])
  
  const [selectedTenant, setSelectedTenant] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  
  // Dummy features based on fraud typical dataset
  const [features, setFeatures] = useState({
    V1: 0.1,
    V2: -0.5,
    V3: 1.2,
    Amount: 100.50
  })

  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState<PredictionRecord[]>([])
  const { addToast } = useToast()

  const session = getSession()

  useEffect(() => {
    if (!session) return
    if (session.role === 'admin') {
      listTenants(session.token).then(res => setTenants(res.tenants)).catch(e => console.error(e))
    } else {
      setSelectedTenant(session.tenant_id as string)
    }
  }, [])

  useEffect(() => {
    if (!selectedTenant) return
    listTenantModels(session!.token, selectedTenant)
      .then(res => {
        setModels(res.models)
        if (res.models.length > 0) setSelectedModel(res.models[0].model_id)
      })
      .catch(e => console.error(e))
  }, [selectedTenant])

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedTenant || !selectedModel) {
      addToast('Select tenant and model first', 'warning')
      return
    }

    setLoading(true)
    const start = performance.now()
    try {
      const res = await predict(session!.token, selectedTenant, selectedModel, features)
      const latency = Math.round(performance.now() - start)
      
      const newRecord = {
        id: Math.random().toString(36).substring(2, 9),
        features: JSON.stringify(features),
        prediction: res.prediction,
        probability: res.probability,
        latency,
        timestamp: new Date().toLocaleTimeString()
      }
      
      setHistory(prev => [newRecord, ...prev].slice(0, 10))
    } catch (err: any) {
      addToast(err.message, 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-white tracking-tight">Inference Playground</h1>
        <p className="text-slate-400 mt-1">Test your live models in real-time.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-medium text-white mb-4">Configuration</h3>
            
            {session?.role === 'admin' && (
              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-300 mb-1">Tenant</label>
                <select 
                  value={selectedTenant}
                  onChange={(e) => setSelectedTenant(e.target.value)}
                  className="w-full bg-[#0f1117] border border-[#2d3348] text-white rounded-lg px-3 py-2 outline-none focus:border-indigo-500"
                >
                  <option value="">-- Select Tenant --</option>
                  {tenants.map(t => <option key={t.tenant_id} value={t.tenant_id}>{t.tenant_name} ({t.tenant_id})</option>)}
                </select>
              </div>
            )}
            
            <div>
              <label className="block text-sm font-medium text-slate-300 mb-1">Model</label>
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-[#0f1117] border border-[#2d3348] text-white rounded-lg px-3 py-2 outline-none focus:border-indigo-500"
                disabled={!selectedTenant}
              >
                <option value="">-- Select Model --</option>
                {models.map(m => <option key={m.model_id} value={m.model_id}>{m.model_id} (v{m.model_version})</option>)}
              </select>
            </div>
          </div>

          <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 shadow-sm">
            <h3 className="text-lg font-medium text-white mb-4">Feature Inputs</h3>
            <form onSubmit={handlePredict} className="space-y-4">
              {Object.entries(features).map(([key, val]) => (
                <div key={key}>
                  <label className="block text-sm font-medium text-slate-300 mb-1">{key}</label>
                  <input 
                    type="number"
                    step="any"
                    value={val}
                    onChange={(e) => setFeatures({...features, [key]: parseFloat(e.target.value)})}
                    className="w-full bg-[#0f1117] border border-[#2d3348] text-white rounded-lg px-3 py-2 outline-none focus:border-indigo-500"
                  />
                </div>
              ))}
              <button
                type="submit"
                disabled={loading || !selectedModel}
                className="w-full mt-2 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white py-2.5 rounded-lg font-medium transition-colors"
              >
                {loading ? 'Predicting...' : 'Predict'}
              </button>
            </form>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-6">
          <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl overflow-hidden">
            <div className="p-6 border-b border-[#2d3348]">
              <h3 className="text-lg font-medium text-white">Recent Predictions</h3>
            </div>
            {history.length === 0 ? (
              <div className="p-8 text-center text-slate-400">No predictions yet. Submit a request to see results.</div>
            ) : (
              <table className="w-full text-left text-sm">
                <thead className="bg-[#0f1117]/50 border-b border-[#2d3348] text-slate-400">
                  <tr>
                    <th className="px-6 py-4 font-medium">Time</th>
                    <th className="px-6 py-4 font-medium">Features</th>
                    <th className="px-6 py-4 font-medium">Prediction</th>
                    <th className="px-6 py-4 font-medium">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#2d3348]">
                  {history.map((record) => (
                    <tr key={record.id} className="hover:bg-[#2d3348]/20">
                      <td className="px-6 py-4 text-slate-400">{record.timestamp}</td>
                      <td className="px-6 py-4 font-mono text-xs text-slate-400 max-w-xs truncate" title={record.features}>
                        {record.features}
                      </td>
                      <td className="px-6 py-4">
                        <span className={`px-2 py-1 rounded text-xs font-semibold ${record.prediction === 1 ? 'bg-red-500/10 text-red-400' : 'bg-green-500/10 text-green-400'}`}>
                          {record.prediction === 1 ? 'FRAUD' : 'LEGIT'} ({record.probability.toFixed(3)})
                        </span>
                      </td>
                      <td className="px-6 py-4 text-slate-400">{record.latency} ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
