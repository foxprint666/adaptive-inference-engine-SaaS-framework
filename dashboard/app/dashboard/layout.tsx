'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { getSession, clearSession, Session } from '@/lib/auth'
import { SidebarNav } from '@/components/nav'
import { useToast } from '@/components/toast'

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [session, setSession] = useState<Session | null>(null)
  const [sseConnected, setSseConnected] = useState(false)
  const router = useRouter()
  const { addToast } = useToast()

  useEffect(() => {
    const s = getSession()
    if (!s) {
      router.push('/login')
    } else {
      setSession(s)
      
      // Setup SSE if admin
      if (s.role === 'admin') {
        const ADMIN_URL = process.env.NEXT_PUBLIC_ADMIN_URL || 'http://localhost:8003'
        const eventSource = new EventSource(`${ADMIN_URL}/events/stream`)
        
        eventSource.onopen = () => setSseConnected(true)
        eventSource.onerror = () => setSseConnected(false)
        
        eventSource.addEventListener('model_reload', (e) => {
          try {
            const data = JSON.parse(e.data)
            addToast(`Model Reloaded: ${data.model_id} (Tenant: ${data.tenant_id})`, 'info')
          } catch (err) {}
        })

        return () => eventSource.close()
      }
    }
  }, [router, addToast])

  const handleLogout = () => {
    clearSession()
    router.push('/login')
  }

  if (!session) return <div className="min-h-screen bg-[#0f1117] flex items-center justify-center">Loading...</div>

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-[#0f1117]">
      {/* Top Header */}
      <header className="h-16 bg-[#1a1f2e] border-b border-[#2d3348] flex items-center justify-between px-6 shrink-0">
        <div className="flex items-center gap-4">
          <div className="font-semibold text-lg tracking-tight text-white flex items-center gap-2">
            <div className="w-6 h-6 bg-indigo-500 rounded-md flex items-center justify-center text-xs">AI</div>
            MLOps Platform
          </div>
          <div className="h-4 w-px bg-[#2d3348]" />
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <div className={`w-2 h-2 rounded-full ${sseConnected ? 'bg-green-500' : 'bg-slate-600'}`} />
            {sseConnected ? 'Live' : 'Disconnected'}
          </div>
        </div>

        <div className="flex items-center gap-4 text-sm">
          <div className="flex flex-col items-end">
            <span className="text-slate-200 font-medium">{session.display_name}</span>
            <span className="text-xs text-indigo-400 uppercase tracking-wider font-semibold">{session.role}</span>
          </div>
          <button 
            onClick={handleLogout}
            className="px-3 py-1.5 rounded bg-[#2d3348] hover:bg-slate-700 text-slate-200 transition-colors"
          >
            Logout
          </button>
        </div>
      </header>

      {/* Main Layout */}
      <div className="flex flex-1 overflow-hidden">
        <SidebarNav />
        <main className="flex-1 overflow-y-auto p-8">
          {children}
        </main>
      </div>
    </div>
  )
}
