'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import { listTenants, registerTenant, Tenant } from '@/lib/api'
import { getSession } from '@/lib/auth'
import { useToast } from '@/components/toast'
import { useRouter } from 'next/navigation'

export default function TenantsPage() {
  const [tenants, setTenants] = useState<Tenant[]>([])
  const [loading, setLoading] = useState(true)
  const [showModal, setShowModal] = useState(false)
  const [formData, setFormData] = useState({
    tenant_id: '',
    tenant_name: '',
    contact_email: '',
    tier: 'standard',
  })
  const { addToast } = useToast()
  const router = useRouter()
  const session = getSession()

  useEffect(() => {
    if (session?.role !== 'admin') {
      if (session?.tenant_id) {
        router.push(`/dashboard/tenants/${session.tenant_id}`)
      }
      return
    }
    loadTenants()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const loadTenants = async () => {
    try {
      const res = await listTenants(session!.token)
      setTenants(res.tenants)
    } catch (err: unknown) {
      addToast((err as { message?: string }).message || 'Failed to load tenants', 'error')
    } finally {
      setLoading(false)
    }
  }

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await registerTenant(session!.token, {
        tenant_id: formData.tenant_id,
        tenant_name: formData.tenant_name,
        contact_email: formData.contact_email,
        tier: formData.tier,
      })
      addToast('Tenant registered successfully', 'success')
      setShowModal(false)
      setFormData({ tenant_id: '', tenant_name: '', contact_email: '', tier: 'standard' })
      loadTenants()
    } catch (err: unknown) {
      addToast((err as { message?: string }).message || 'Registration failed', 'error')
    }
  }

  if (session?.role !== 'admin') return null

  return (
    <div className="max-w-6xl space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-white tracking-tight">Tenants</h1>
          <p className="text-slate-400 mt-1">Manage platform tenants and their model registries.</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
        >
          <span>+</span> Add Tenant
        </button>
      </div>

      <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl overflow-hidden">
        {loading ? (
          <div className="divide-y divide-[#2d3348]">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="px-6 py-4 flex gap-6">
                <div className="h-4 w-28 bg-[#2d3348] rounded animate-pulse" />
                <div className="h-4 w-40 bg-[#2d3348] rounded animate-pulse" />
                <div className="h-4 w-20 bg-[#2d3348] rounded animate-pulse" />
              </div>
            ))}
          </div>
        ) : (
          <table className="w-full text-left text-sm">
            <thead className="bg-[#0f1117]/50 border-b border-[#2d3348] text-slate-400">
              <tr>
                <th className="px-6 py-4 font-medium">Tenant ID</th>
                <th className="px-6 py-4 font-medium">Name</th>
                <th className="px-6 py-4 font-medium">Tier</th>
                <th className="px-6 py-4 font-medium">Contact</th>
                <th className="px-6 py-4 font-medium">Models</th>
                <th className="px-6 py-4 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#2d3348]">
              {tenants.map(tenant => (
                <tr key={tenant.tenant_id} className="hover:bg-[#2d3348]/20 transition-colors">
                  <td className="px-6 py-4 font-mono text-xs text-slate-300">{tenant.tenant_id}</td>
                  <td className="px-6 py-4 font-medium text-slate-200">{tenant.tenant_name}</td>
                  <td className="px-6 py-4">
                    <span className="px-2 py-1 bg-indigo-500/10 text-indigo-400 rounded text-xs uppercase tracking-wider font-semibold">
                      {tenant.tier}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-slate-400">{tenant.contact_email}</td>
                  <td className="px-6 py-4 text-slate-400">{tenant.model_count}</td>
                  <td className="px-6 py-4">
                    <Link href={`/dashboard/tenants/${tenant.tenant_id}`} className="text-indigo-400 hover:text-indigo-300 font-medium">
                      Details →
                    </Link>
                  </td>
                </tr>
              ))}
              {tenants.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-6 py-12 text-center text-slate-400">
                    <div className="text-3xl mb-2">🏢</div>
                    No tenants registered yet. Click <strong>Add Tenant</strong> to get started.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>

      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1a1f2e] border border-[#2d3348] rounded-xl p-6 w-full max-w-md shadow-2xl">
            <h2 className="text-xl font-semibold text-white mb-5">Register New Tenant</h2>
            <form onSubmit={handleRegister} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Tenant ID <span className="text-slate-500">(slug, e.g. bank-a)</span></label>
                <input
                  required
                  type="text"
                  pattern="[a-z0-9-]+"
                  title="Lowercase letters, numbers, hyphens only"
                  value={formData.tenant_id}
                  onChange={e => setFormData({ ...formData, tenant_id: e.target.value })}
                  className="w-full bg-[#0f1117] border border-[#2d3348] text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 outline-none font-mono text-sm"
                  placeholder="bank-a"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Organisation Name</label>
                <input
                  required
                  type="text"
                  value={formData.tenant_name}
                  onChange={e => setFormData({ ...formData, tenant_name: e.target.value })}
                  className="w-full bg-[#0f1117] border border-[#2d3348] text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 outline-none"
                  placeholder="First National Bank"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Admin Email</label>
                <input
                  required
                  type="email"
                  value={formData.contact_email}
                  onChange={e => setFormData({ ...formData, contact_email: e.target.value })}
                  className="w-full bg-[#0f1117] border border-[#2d3348] text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 outline-none"
                  placeholder="admin@bank-a.com"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-1">Tier</label>
                <select
                  value={formData.tier}
                  onChange={e => setFormData({ ...formData, tier: e.target.value })}
                  className="w-full bg-[#0f1117] border border-[#2d3348] text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-indigo-500 outline-none"
                >
                  <option value="free">Free</option>
                  <option value="standard">Standard</option>
                  <option value="enterprise">Enterprise</option>
                </select>
              </div>
              <div className="flex gap-3 justify-end mt-6 pt-4 border-t border-[#2d3348]">
                <button
                  type="button"
                  onClick={() => setShowModal(false)}
                  className="px-4 py-2 text-slate-300 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="bg-indigo-600 hover:bg-indigo-700 text-white px-5 py-2 rounded-lg font-medium transition-colors"
                >
                  Register
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
