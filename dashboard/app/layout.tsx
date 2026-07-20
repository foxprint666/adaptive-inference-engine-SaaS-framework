import type { Metadata } from 'next'
import { Geist } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/providers'

const geist = Geist({
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'MLOps Dashboard',
  description: 'Adaptive Inference Engine SaaS Platform',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${geist.className} antialiased bg-[#0f1117] text-[#f1f5f9] min-h-screen`}>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  )
}
