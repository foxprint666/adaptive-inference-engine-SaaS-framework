'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { clsx } from 'clsx'
import { Activity, Database, LayoutDashboard, Play, Settings, ShieldAlert } from 'lucide-react'

const navItems = [
  { name: 'Overview', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Tenants', href: '/dashboard/tenants', icon: Database },
  { name: 'Models', href: '/dashboard/models', icon: Activity },
  { name: 'Drift & Retraining', href: '/dashboard/drift', icon: ShieldAlert },
  { name: 'Playground', href: '/dashboard/playground', icon: Play },
  { name: 'System Health', href: '/dashboard/health', icon: Settings },
]

export function SidebarNav() {
  const pathname = usePathname()

  return (
    <nav className="flex flex-col gap-1 w-64 border-r border-[#2d3348] bg-[#1a1f2e] h-[calc(100vh-64px)] p-4">
      {navItems.map((item) => {
        const isActive = pathname === item.href || (item.href !== '/dashboard' && pathname.startsWith(item.href))
        const Icon = item.icon
        
        return (
          <Link
            key={item.name}
            href={item.href}
            className={clsx(
              'flex items-center gap-3 px-3 py-2.5 rounded-md text-sm font-medium transition-colors',
              isActive
                ? 'bg-indigo-500/10 text-indigo-400'
                : 'text-slate-400 hover:bg-[#2d3348]/50 hover:text-slate-200'
            )}
          >
            <Icon className="w-4 h-4" />
            {item.name}
          </Link>
        )
      })}
    </nav>
  )
}
