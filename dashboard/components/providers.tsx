'use client'

import { ToastProvider } from './toast'
import { ReactNode } from 'react'

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ToastProvider>
      {children}
    </ToastProvider>
  )
}
