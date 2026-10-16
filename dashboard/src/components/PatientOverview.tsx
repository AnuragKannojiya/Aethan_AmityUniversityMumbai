'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  User,
  MapPin,
  Calendar,
  Activity,
  Pill,
  ShieldAlert,
  ArrowUp,
  ArrowUpRight,
  ArrowRight,
  ArrowDownRight,
  ArrowDown
} from 'lucide-react';
import { Patient } from '../lib/types';
import { getGlucoseColor, getGlucoseBadgeClass, getRiskBadgeClasses } from '../lib/utils';

interface PatientOverviewProps {
  patient: Patient;
}

export default function PatientOverview({ patient }: PatientOverviewProps) {
  // Initials for avatar
  const initials = patient.name
    .split(' ')
    .filter(Boolean)
    .map((n) => n[0])
    .join('')
    .substring(0, 2)
    .toUpperCase();

  const glucoseBadge = getGlucoseBadgeClass(patient.lastGlucose);
  const glucoseHex = getGlucoseColor(patient.lastGlucose);

  const renderTrendIcon = () => {
    switch (patient.glucoseTrend) {
      case 'rising_fast':
        return (
          <span className="flex items-center text-red-400 font-bold text-lg animate-pulse" title="Rising fast (>2 mg/dL/min)">
            <ArrowUp className="w-5 h-5 -mr-1" />
            <ArrowUp className="w-5 h-5" />
          </span>
        );
      case 'rising':
        return (
          <span className="flex items-center text-amber-400 font-bold text-lg" title="Rising (1-2 mg/dL/min)">
            <ArrowUpRight className="w-5 h-5" />
          </span>
        );
      case 'stable':
        return (
          <span className="flex items-center text-emerald-400 font-bold text-lg" title="Stable (±1 mg/dL/min)">
            <ArrowRight className="w-5 h-5" />
          </span>
        );
      case 'falling':
        return (
          <span className="flex items-center text-cyan-400 font-bold text-lg" title="Falling (1-2 mg/dL/min)">
            <ArrowDownRight className="w-5 h-5" />
          </span>
        );
      case 'falling_fast':
        return (
          <span className="flex items-center text-yellow-400 font-bold text-lg" title="Falling fast (>2 mg/dL/min)">
            <ArrowDown className="w-5 h-5 -mr-1" />
            <ArrowDown className="w-5 h-5" />
          </span>
        );
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="bg-[#111827] border border-[#1f293d] rounded-xl p-5 shadow-lg flex flex-col justify-between h-full"
      aria-label="Patient Clinical Overview"
    >
      <div>
        {/* Header: Avatar, Name, Demographics, and Current Glucose */}
        <div className="flex items-start justify-between pb-4 border-b border-[#1f293d]">
          <div className="flex items-center gap-3.5">
            <div
              className="w-13 h-13 rounded-full flex items-center justify-center font-bold text-base shadow-inner ring-2"
              style={{
                backgroundColor: 'rgba(6, 182, 212, 0.15)',
                color: '#06b6d4',
                borderColor: glucoseHex,
              }}
            >
              {initials || 'PT'}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold text-white tracking-tight">
                  {patient.name}
                </h2>
                <span className="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                  {patient.id}
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-400 mt-1">
                <span>{patient.age} yrs • {patient.gender}</span>
                <span className="flex items-center gap-1">
                  <MapPin className="w-3.5 h-3.5 text-cyan-500" />
                  {patient.city}
                </span>
              </div>
            </div>
          </div>

          {/* Current Glucose Callout */}
          <div className="text-right">
            <span className="text-[10px] uppercase tracking-wider text-slate-400 font-medium">
              Current CGM
            </span>
            <div className="flex items-baseline justify-end gap-1.5 mt-0.5">
              <span
                className="text-2xl font-bold tracking-tight"
                style={{ color: glucoseHex }}
              >
                {patient.lastGlucose}
              </span>
              <span className="text-xs text-slate-400 font-medium">mg/dL</span>
              <div className="ml-1">{renderTrendIcon()}</div>
            </div>
            <span
              className={`inline-block text-[10px] px-2 py-0.5 rounded-full font-medium border ${glucoseBadge.bg} ${glucoseBadge.text} ${glucoseBadge.border}`}
            >
              {glucoseBadge.label}
            </span>
          </div>
        </div>

        {/* Risk Level Bar */}
        <div className="mt-4 pt-1">
          <div className="flex justify-between items-center text-xs mb-1.5">
            <span className="text-slate-400 font-medium flex items-center gap-1.5">
              <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
              Dynamic Twin Risk Index:
            </span>
            <span
              className={`px-2 py-0.5 text-[11px] font-semibold rounded ${getRiskBadgeClasses(
                patient.currentRisk
              )}`}
            >
              {patient.currentRisk.toUpperCase()} ({patient.currentRiskScore}%)
            </span>
          </div>
          {/* Segmented Risk Bar */}
          <div className="w-full bg-slate-800/80 rounded-full h-2 overflow-hidden flex">
            <div
              className="h-full rounded-full transition-all duration-700 ease-out"
              style={{
                width: `${patient.currentRiskScore}%`,
                backgroundColor:
                  patient.currentRisk === 'Critical'
                    ? '#ef4444'
                    : patient.currentRisk === 'High'
                    ? '#f97316'
                    : patient.currentRisk === 'Moderate'
                    ? '#f59e0b'
                    : '#22c55e',
              }}
            />
          </div>
        </div>

        {/* Key Vitals Grid */}
        <div className="grid grid-cols-3 gap-2.5 mt-4">
          <div className="bg-[#0a0e1a]/80 border border-[#1f293d]/80 rounded-lg p-2.5 text-center">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">
              HbA1c
            </span>
            <span
              className={`text-lg font-bold block mt-0.5 ${
                patient.hba1c >= 8.0
                  ? 'text-red-400'
                  : patient.hba1c >= 7.0
                  ? 'text-amber-400'
                  : 'text-emerald-400'
              }`}
            >
              {patient.hba1c.toFixed(1)}%
            </span>
            <span className="text-[10px] text-slate-500">
              {patient.hba1c >= 8.0 ? 'Suboptimal' : patient.hba1c >= 7.0 ? 'Borderline' : 'Controlled'}
            </span>
          </div>

          <div className="bg-[#0a0e1a]/80 border border-[#1f293d]/80 rounded-lg p-2.5 text-center">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">
              BMI
            </span>
            <span className="text-lg font-bold text-slate-200 block mt-0.5">
              {patient.bmi.toFixed(1)}
            </span>
            <span className="text-[10px] text-slate-500">
              {patient.bmi >= 27.5 ? 'Obese (Asian)' : 'Overweight'}
            </span>
          </div>

          <div className="bg-[#0a0e1a]/80 border border-[#1f293d]/80 rounded-lg p-2.5 text-center">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-medium">
              T2D History
            </span>
            <span className="text-lg font-bold text-slate-200 block mt-0.5">
              {patient.diabetesDurationYears}
            </span>
            <span className="text-[10px] text-slate-500">years duration</span>
          </div>
        </div>

        {/* Comorbidities (Past Diagnoses) */}
        <div className="mt-4">
          <span className="text-xs font-semibold text-slate-300 block mb-2 flex items-center gap-1.5">
            <Activity className="w-3.5 h-3.5 text-cyan-400" />
            Past Diagnoses & Comorbidities (ICD-10)
          </span>
          <div className="flex flex-wrap gap-1.5">
            {patient.comorbidities.map((item, idx) => (
              <span
                key={idx}
                className={`text-[11px] px-2.5 py-1 rounded-md font-medium border ${
                  item.severity === 'severe'
                    ? 'bg-red-950/40 text-red-300 border-red-800/40'
                    : item.severity === 'moderate'
                    ? 'bg-amber-950/40 text-amber-300 border-amber-800/40'
                    : 'bg-slate-800 text-slate-300 border-slate-700/60'
                }`}
              >
                {item.name}{item.icd10 ? ` (${item.icd10})` : ''}
              </span>
            ))}
          </div>
        </div>

        {/* EHR Lab Results & Genetic Markers */}
        <div className="mt-4 pt-3 border-t border-[#1f293d]/80 grid grid-cols-2 gap-2.5">
          <div className="bg-[#0a0e1a]/70 border border-[#1f293d]/70 rounded-lg p-2.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-cyan-400 block mb-1.5">
              EHR Lab Results
            </span>
            <div className="space-y-1 text-[11px]">
              <div className="flex justify-between">
                <span className="text-slate-400">FastingBG:</span>
                <span className="font-mono text-slate-200">{Math.round(patient.hba1c * 24 - 35)} mg/dL</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">BP / eGFR:</span>
                <span className="font-mono text-slate-200">
                  {Math.round(118 + patient.age * 0.4)}/{Math.round(76 + patient.bmi * 0.3)} • {Math.max(42, Math.round(112 - patient.age * 0.6))}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">TG / LDL:</span>
                <span className="font-mono text-slate-200">
                  {Math.round(115 + patient.bmi * 2.8)} / {Math.round(88 + patient.hba1c * 4.5)}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-[#0a0e1a]/70 border border-[#1f293d]/70 rounded-lg p-2.5">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-purple-400 block mb-1.5">
              Genetic Markers (SNPs)
            </span>
            <div className="space-y-1 text-[11px]">
              <div className="flex justify-between">
                <span className="text-slate-400">TCF7L2:</span>
                <span className="font-mono text-amber-300">
                  {patient.hba1c >= 8.5 ? 'TT (High Risk)' : patient.hba1c >= 7.5 ? 'CT (Het Risk)' : 'CC (Wild)'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">KCNJ11 / SLC30A8:</span>
                <span className="font-mono text-slate-200">
                  {patient.hba1c >= 8.0 ? 'TT / CC' : 'CT / CT'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Polygenic Score:</span>
                <span className="font-mono text-purple-300 font-semibold">
                  {Math.min(0.94, Math.max(0.22, Number((0.25 + (patient.hba1c - 6.0) * 0.11).toFixed(2))))} PRS
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Medications */}
      <div className="mt-4 pt-3 border-t border-[#1f293d]">
        <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 mb-2">
          <Pill className="w-3.5 h-3.5 text-cyan-400" />
          Active Regimen ({patient.medications.length})
        </span>
        <ul className="space-y-1 text-xs text-slate-300">
          {patient.medications.map((med, idx) => (
            <li
              key={idx}
              className="flex items-center gap-2 bg-[#0a0e1a]/50 px-2 py-1.5 rounded border border-[#1f293d]/40"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 shrink-0" />
              <span className="truncate">{med}</span>
            </li>
          ))}
        </ul>
      </div>
    </motion.section>
  );
}
