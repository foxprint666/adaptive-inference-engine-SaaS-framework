'use client'

import { useEffect, useState } from 'react'
import { useParams, useRouter } from 'next/navigation'
import Link from 'next/link'
import { getTenant, TenantDetail } from '@/lib/api'
import { getSession } from '@/lib/auth'
import { useToast } from '@/components/toast'

export default function TenantDetailPage() {
  const params = useParams()
  const tenantId = params.tenantId as string
  const [tenant, setTenant] = useState<TenantDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const { addToast } = useToast()
  const router = useRouter()

  useEffect(() => {
    const session = getSession()
    if (!session) return

    getTenant(session.token, tenantId)
      .then(setTenant)
      .catch(err => {
        addToast(err.message, 'error')
        router.push('/dashboard/tenants')
      })
      .finally(() => setLoading(false))
  }, [tenantId])

  if (loading) return <div className="animate-pulse space-y-6 max-w-6xl">
    <div className="h-32 bg-[#1a1f2e] rounded-xl border border-[#2d3348]"></div>
    <div className="h-64 bg-[#1a1f2e] rounded-xl border border-[#2d3348]"></div>
  </div>

  if (!tenant) return null

  return (
    <div className="max-w-6xl space-y-6">
      <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 shadow-sm">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-2xl font-semibold text-white tracking-tight">{tenant.tenant_name}</h1>
            <p className="text-slate-400 font-mono text-sm mt-1">{tenant.tenant_id}</p>
          </div>
          <span className="px-3 py-1 bg-indigo-500/10 text-indigo-400 rounded-md text-xs uppercase tracking-wider font-semibold">
            {tenant.tier} Tier
          </span>
        </div>
        <div className="mt-6 flex gap-8 border-t border-[#2d3348] pt-6">
          <div>
            <div className="text-sm text-slate-400">Admin Email</div>
            <div className="text-slate-200 mt-1">{tenant.contact_email}</div>
          </div>
          <div>
            <div className="text-sm text-slate-400">Created At</div>
            <div className="text-slate-200 mt-1">N/A</div>
          </div>
          <div>
            <div className="text-sm text-slate-400">Total Models</div>
            <div className="text-slate-200 mt-1">{tenant.models?.length || 0}</div>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between mt-8">
        <h2 className="text-xl font-semibold text-white">Registered Models</h2>
        <button className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
          Register Model
        </button>
      </div>

      <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl overflow-hidden">
        <table className="w-full text-left text-sm">
          <thead className="bg-[#0f1117]/50 border-b border-[#2d3348] text-slate-400">
            <tr>
              <th className="px-6 py-4 font-medium">Model ID</th>
              <th className="px-6 py-4 font-medium">Version</th>
              <th className="px-6 py-4 font-medium">Framework</th>
              <th className="px-6 py-4 font-medium">Threshold</th>
              <th className="px-6 py-4 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[#2d3348]">
            {tenant.models?.map(model => (
              <tr key={model.model_id} className="hover:bg-[#2d3348]/20 transition-colors">
                <td className="px-6 py-4 font-mono text-xs text-slate-300">{model.model_id}</td>
                <td className="px-6 py-4 text-slate-200">{model.model_version}</td>
                <td className="px-6 py-4 text-slate-400 capitalize">{model.framework}</td>
                <td className="px-6 py-4 text-slate-400">{model.drift_thresholds?.psi_threshold ?? 'Default'}</td>
                <td className="px-6 py-4">
                  <Link href={`/dashboard/drift/${tenant.tenant_id}/${model.model_id}`} className="text-indigo-400 hover:text-indigo-300 font-medium">
                    View Drift Monitor →
                  </Link>
                </td>
              </tr>
            ))}
            {(!tenant.models || tenant.models.length === 0) && (
              <tr>
                <td colSpan={5} className="px-6 py-8 text-center text-slate-400">No models registered for this tenant.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
