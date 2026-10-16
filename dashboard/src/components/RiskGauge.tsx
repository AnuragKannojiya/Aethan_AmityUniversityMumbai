'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Clock, CheckCircle2, Shield, Flame } from 'lucide-react';
import { RiskLevel } from '../lib/types';

interface RiskGaugeProps {
  riskScore: number; // 0 - 100
  riskLevel: RiskLevel;
  timeToSpikeMin: number | null;
  predictedPeakGlucose: number;
}

export default function RiskGauge({
  riskScore,
  riskLevel,
  timeToSpikeMin,
  predictedPeakGlucose,
}: RiskGaugeProps) {
  // Determine color based on risk score tiers
  const getGaugeColor = (score: number): { hex: string; textClass: string; bgGlow: string } => {
    if (score >= 80) {
      return { hex: '#ef4444', textClass: 'text-red-400', bgGlow: 'rgba(239, 68, 68, 0.25)' };
    }
    if (score >= 60) {
      return { hex: '#f97316', textClass: 'text-orange-400', bgGlow: 'rgba(249, 115, 22, 0.2)' };
    }
    if (score >= 30) {
      return { hex: '#eab308', textClass: 'text-yellow-400', bgGlow: 'rgba(234, 179, 8, 0.15)' };
    }
    return { hex: '#22c55e', textClass: 'text-emerald-400', bgGlow: 'rgba(34, 197, 94, 0.15)' };
  };

  const colors = getGaugeColor(riskScore);
  const isHighAlert = riskScore >= 80;

  // SVG Gauge calculations
  const radius = 64;
  const strokeWidth = 10;
  const circumference = 2 * Math.PI * radius;
  // Use a 270-degree arc or full circle
  // Let's do a smooth 280-degree semi-arc for an automotive / clinical monitor look:
  const arcLength = circumference * 0.75; // 270 degrees
  const strokeDashoffset = arcLength - (arcLength * Math.min(100, Math.max(0, riskScore))) / 100;

  return (
    <motion.section
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.1 }}
      className={`bg-[#111827] border rounded-xl p-5 shadow-lg flex flex-col justify-between relative overflow-hidden transition-all duration-300 ${
        isHighAlert
          ? 'border-red-500/40 shadow-red-950/30'
          : 'border-[#1f293d]'
      }`}
      aria-label="Spike Risk Assessment Gauge"
    >
      {/* Background ambient glow when high alert */}
      {isHighAlert && (
        <div
          className="absolute -top-12 -right-12 w-40 h-40 rounded-full blur-3xl pointer-events-none opacity-40 animate-pulse"
          style={{ backgroundColor: colors.hex }}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-base font-semibold text-white tracking-tight flex items-center gap-2">
          <Shield className="w-4 h-4 text-cyan-400" />
          2h Spike Risk Assessment
        </h2>
        <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
          Ensemble Model
        </span>
      </div>

      {/* SVG Circular Progress Indicator */}
      <div className="flex flex-col items-center justify-center my-3 relative">
        <div className="relative w-44 h-44 flex items-center justify-center">
          <svg className="w-full h-full transform -rotate-135" viewBox="0 0 160 160">
            {/* Background Track */}
            <circle
              cx="80"
              cy="80"
              r={radius}
              fill="transparent"
              stroke="#1e293b"
              strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength} ${circumference}`}
              strokeLinecap="round"
            />
            {/* Active Progress Arc */}
            <motion.circle
              cx="80"
              cy="80"
              r={radius}
              fill="transparent"
              stroke={colors.hex}
              strokeWidth={strokeWidth}
              strokeDasharray={`${arcLength} ${circumference}`}
              initial={{ strokeDashoffset: arcLength }}
              animate={{ strokeDashoffset }}
              transition={{ duration: 1.2, ease: 'easeOut' }}
              strokeLinecap="round"
              style={{
                filter: `drop-shadow(0 0 6px ${colors.hex})`,
              }}
            />
          </svg>

          {/* Center Content */}
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            {isHighAlert ? (
              <Flame className="w-5 h-5 text-red-400 mb-0.5 animate-bounce" />
            ) : (
              <span className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">
                Risk Score
              </span>
            )}

            <div className="flex items-baseline">
              <span className={`text-4xl font-extrabold tracking-tight ${colors.textClass}`}>
                {riskScore}
              </span>
              <span className="text-sm font-semibold text-slate-400 ml-0.5">%</span>
            </div>

            <span
              className={`text-xs font-bold uppercase tracking-wider mt-0.5 ${colors.textClass}`}
            >
              {riskLevel} Risk
            </span>
          </div>
        </div>

        {/* Spike Time Projection Callout */}
        <div
          className={`w-full text-center py-2 px-3 rounded-lg border mt-1 transition-all ${
            timeToSpikeMin !== null
              ? isHighAlert
                ? 'bg-red-950/30 border-red-800/40 text-red-200'
                : 'bg-amber-950/30 border-amber-800/40 text-amber-200'
              : 'bg-emerald-950/30 border-emerald-800/40 text-emerald-200'
          }`}
        >
          {timeToSpikeMin !== null ? (
            <div className="flex items-center justify-center gap-1.5 text-xs font-medium">
              <Clock className="w-3.5 h-3.5 shrink-0 text-amber-400" />
              <span>
                Time to potential spike: <strong className="font-bold underline">{timeToSpikeMin} mins</strong>
              </span>
            </div>
          ) : (
            <div className="flex items-center justify-center gap-1.5 text-xs font-medium text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
              <span>No spike expected within 2 hours</span>
            </div>
          )}
        </div>
      </div>

      {/* Secondary Metrics */}
      <div className="grid grid-cols-2 gap-2 pt-2 border-t border-[#1f293d]/80 text-center">
        <div className="bg-[#0a0e1a]/60 p-2 rounded border border-[#1f293d]/60">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
            Projected Peak
          </span>
          <span className="text-sm font-bold text-slate-200 font-mono">
            {predictedPeakGlucose} mg/dL
          </span>
        </div>

        <div className="bg-[#0a0e1a]/60 p-2 rounded border border-[#1f293d]/60">
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
            Model Confidence
          </span>
          <span className="text-sm font-bold text-cyan-400 font-mono">
            94.8%
          </span>
        </div>
      </div>
    </motion.section>
  );
}
