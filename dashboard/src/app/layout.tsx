import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'GlucoTwin | Digital Twin Dashboard',
  description:
    'Real-time physiological digital twin forecasting glucose excursions 2 hours in advance using multimodal EHR and wearable CGM telemetry.',
  keywords: [
    'Digital Twin',
    'Type 2 Diabetes',
    'Glucose Spike Prediction',
    'Continuous Glucose Monitor',
    'Clinical Decision Support',
    'Amity University Mumbai',
    'Team Aethan'
  ],
  authors: [{ name: 'Team Aethan - Amity University Mumbai' }],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} dark`}>
      <body className="min-h-screen bg-[#0a0e1a] text-slate-100 antialiased selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
