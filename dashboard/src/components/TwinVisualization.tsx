'use client';

/**
 * GlucoTwin - Digital Twin State Visualization & In Silico What-If Simulator
 * Multimodal Physiological State Modeling & Counter-Regulatory Intervention Trial
 * Amity University Mumbai - Team Aethan
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
  CartesianGrid
} from 'recharts';
import {
  Cpu,
  Sparkles,
  Heart,
  Activity,
  Moon,
  Zap,
  TrendingDown,
  TrendingUp,
  Pill,
  Footprints,
  Utensils,
  X,
  Check,
  ShieldCheck,
  AlertTriangle,
  Info,
  Clock,
  ArrowRight,
  Flame,
  Layers,
  ChevronRight
} from 'lucide-react';
import { DigitalTwinState, GlucoseReading, InterventionType, InterventionSimulation } from '../lib/types';
import { getGlucoseColor } from '../lib/utils';
import { simulateIntervention } from '../lib/api';

interface TwinVisualizationProps {
  twinState: DigitalTwinState;
  patientId: string;
  patientName: string;
  historicalReadings?: GlucoseReading[];
}

export default function TwinVisualization({
  twinState,
  patientId,
  patientName,
  historicalReadings = []
}: TwinVisualizationProps) {
  // Modal state
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [selectedIntervention, setSelectedIntervention] = useState<InterventionType>('go_for_walk');
  const [simulationData, setSimulationData] = useState<InterventionSimulation | null>(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [isApplied, setIsApplied] = useState(false);

  // Extract recent glucose readings for sparkline
  const sparklineData = useMemo(() => {
    if (historicalReadings.length > 0) {
      return historicalReadings.slice(-12).map((r) => ({
        time: r.displayTime,
        glucose: r.glucose
      }));
    }
    // Fallback sparkline if no readings
    const base = twinState.currentGlucose;
    return [
      { time: '-120m', glucose: Math.max(70, base - 18) },
      { time: '-90m', glucose: Math.max(70, base - 10) },
      { time: '-60m', glucose: Math.max(70, base - 4) },
      { time: '-30m', glucose: Math.max(70, base + 8) },
      { time: '-15m', glucose: Math.max(70, base + 14) },
      { time: 'Now', glucose: base }
    ];
  }, [historicalReadings, twinState.currentGlucose]);

  // Generate SVG points for mini sparkline
  const svgSparklinePath = useMemo(() => {
    if (sparklineData.length < 2) return '';
    const minG = Math.min(...sparklineData.map((d) => d.glucose), 70);
    const maxG = Math.max(...sparklineData.map((d) => d.glucose), 240);
    const range = Math.max(maxG - minG, 1);
    const width = 140;
    const height = 40;
    const padding = 4;

    const points = sparklineData.map((d, index) => {
      const x = padding + (index / (sparklineData.length - 1)) * (width - 2 * padding);
      const y = height - padding - ((d.glucose - minG) / range) * (height - 2 * padding);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    return points.join(' ');
  }, [sparklineData]);

  // Load simulation data when modal opens or intervention changes
  useEffect(() => {
    let isCancelled = false;

    async function runSim() {
      if (!isModalOpen) return;
      setIsSimulating(true);
      setIsApplied(false);
      try {
        const result = await simulateIntervention(patientId, selectedIntervention);
        if (!isCancelled) {
          setSimulationData(result);
        }
      } catch (err) {
        console.error('Simulation error:', err);
      } finally {
        if (!isCancelled) {
          setIsSimulating(false);
        }
      }
    }

    runSim();
    return () => {
      isCancelled = true;
    };
  }, [isModalOpen, selectedIntervention, patientId]);

  // Prepare combined trajectory data for Recharts in the modal
  const trajectoryChartData = useMemo(() => {
    if (!simulationData) return [];
    return simulationData.baselineTrajectory.map((bPt, idx) => {
      const sPt = simulationData.simulatedTrajectory[idx] || bPt;
      const reduction = Math.max(0, bPt.glucose - sPt.glucose);
      return {
        time: bPt.time,
        baseline: bPt.glucose,
        simulated: sPt.glucose,
        reduction,
        difference: bPt.glucose - sPt.glucose
      };
    });
  }, [simulationData]);

  const glucoseColor = getGlucoseColor(twinState.currentGlucose);
  const isHighStrain =
    twinState.organStress.pancreasBetaCellLoad > 75 ||
    twinState.organStress.hepaticGlucoseProduction > 70;

  return (
    <>
      {/* Digital Twin State Card */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.45, delay: 0.2 }}
        className="bg-[#111827] border border-[#1f293d] rounded-xl p-5 shadow-lg flex flex-col justify-between relative overflow-hidden"
      >
        {/* Ambient background decoration */}
        <div className="absolute top-0 right-0 w-80 h-80 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />

        <div>
          {/* Card Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-[#1f293d]">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400">
                <Cpu className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-base font-semibold text-white tracking-wide">
                    Digital Twin Physiological State
                  </h3>
                  <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30 text-[11px] font-medium text-emerald-400">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                    Live In-Silico Twin
                  </span>
                </div>
                <p className="text-xs text-slate-400">
                  Continuous multi-compartment metabolic telemetry for {patientName}
                </p>
              </div>
            </div>

            {/* Run What-If Simulation Trigger Button */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => setIsModalOpen(true)}
              className="px-3.5 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 font-semibold text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
            >
              <Sparkles className="w-4 h-4 fill-slate-950" />
              <span>What-If Simulation</span>
            </motion.button>
          </div>

          {/* 4 Core Physiological Telemetry Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5 my-4">
            {/* Tile 1: Glycemic Dynamics & Sparkline */}
            <div className="bg-[#0b1120] border border-[#1e293b] rounded-lg p-3 flex flex-col justify-between relative group hover:border-cyan-500/40 transition-colors">
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                    Glycemic Flux
                  </span>
                  <div className="flex items-baseline gap-1.5 mt-0.5">
                    <span className="text-2xl font-bold" style={{ color: glucoseColor }}>
                      {twinState.currentGlucose}
                    </span>
                    <span className="text-[11px] text-slate-400">mg/dL</span>
                  </div>
                </div>

                <div className="text-right">
                  <span
                    className={`inline-flex items-center gap-0.5 text-xs font-semibold px-2 py-0.5 rounded ${
                      twinState.glucoseTrendRate > 0
                        ? 'bg-red-500/15 text-red-400'
                        : 'bg-emerald-500/15 text-emerald-400'
                    }`}
                  >
                    {twinState.glucoseTrendRate > 0 ? (
                      <TrendingUp className="w-3.5 h-3.5" />
                    ) : (
                      <TrendingDown className="w-3.5 h-3.5" />
                    )}
                    {twinState.glucoseTrendRate > 0 ? '+' : ''}
                    {twinState.glucoseTrendRate.toFixed(1)}/m
                  </span>
                </div>
              </div>

              {/* Mini Sparkline SVG */}
              <div className="mt-2 pt-2 border-t border-[#1e293b]/60 flex items-center justify-between">
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-400">COB: <strong className="text-slate-200">{twinState.carbsOnBoard}g</strong></span>
                  <span className="text-[10px] text-slate-400">IOB: <strong className="text-slate-200">{twinState.insulinOnBoard}U</strong></span>
                </div>

                <div className="relative w-28 h-9 overflow-hidden">
                  <svg viewBox="0 0 140 40" className="w-full h-full overflow-visible">
                    <defs>
                      <linearGradient id="sparklineGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.4" />
                        <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
                      </linearGradient>
                    </defs>
                    {svgSparklinePath && (
                      <>
                        <polyline
                          fill="none"
                          stroke="#06b6d4"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          points={svgSparklinePath}
                        />
                        {/* Glow dot on latest point */}
                        {sparklineData.length > 0 && (
                          <circle
                            cx="136"
                            cy={
                              40 -
                              4 -
                              ((twinState.currentGlucose -
                                Math.min(...sparklineData.map((d) => d.glucose), 70)) /
                                Math.max(
                                  Math.max(...sparklineData.map((d) => d.glucose), 240) -
                                    Math.min(...sparklineData.map((d) => d.glucose), 70),
                                  1
                                )) *
                                32
                            }
                            r="3.5"
                            fill="#06b6d4"
                            className="animate-ping"
                          />
                        )}
                      </>
                    )}
                  </svg>
                </div>
              </div>
            </div>

            {/* Tile 2: Cardiovascular & Autonomic Tone */}
            <div className="bg-[#0b1120] border border-[#1e293b] rounded-lg p-3 flex flex-col justify-between hover:border-red-500/40 transition-colors">
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                    Cardiovascular Tone
                  </span>
                  <div className="flex items-baseline gap-1.5 mt-0.5">
                    <span className="text-2xl font-bold text-white">{twinState.heartRate}</span>
                    <span className="text-[11px] text-slate-400">bpm</span>
                  </div>
                </div>

                <div className="w-8 h-8 rounded-full bg-red-500/10 border border-red-500/25 flex items-center justify-center text-red-400">
                  <Heart className="w-4 h-4 fill-red-400/30 animate-pulse" />
                </div>
              </div>

              <div className="mt-2 pt-2 border-t border-[#1e293b]/60 flex items-center justify-between text-[11px]">
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-400">Autonomic HRV</span>
                  <span className="font-semibold text-slate-200">{twinState.hrv} ms (RMSSD)</span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-slate-400">CV Strain</span>
                  <span className="font-semibold text-amber-400">
                    {twinState.organStress.cardiovascularStrain}%
                  </span>
                </div>
              </div>
            </div>

            {/* Tile 3: Physical Activity & Metabolic State */}
            <div className="bg-[#0b1120] border border-[#1e293b] rounded-lg p-3 flex flex-col justify-between hover:border-amber-500/40 transition-colors">
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                    Activity & Metabolism
                  </span>
                  <div className="flex items-baseline gap-1.5 mt-0.5">
                    <span className="text-sm font-bold text-cyan-300">
                      {twinState.activityStatus}
                    </span>
                  </div>
                </div>

                <div className="w-8 h-8 rounded-full bg-amber-500/10 border border-amber-500/25 flex items-center justify-center text-amber-400">
                  <Activity className="w-4 h-4" />
                </div>
              </div>

              <div className="mt-2 pt-2 border-t border-[#1e293b]/60 flex items-center justify-between text-[11px]">
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-400">Metabolic Mode</span>
                  <span className="font-semibold text-emerald-400">
                    {twinState.metabolicState}
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-slate-400">Insulin Sensitivity</span>
                  <span className="font-semibold text-slate-200">
                    {twinState.insulinSensitivityFactor} mg/dL/U
                  </span>
                </div>
              </div>
            </div>

            {/* Tile 4: Circadian & Sleep Quality */}
            <div className="bg-[#0b1120] border border-[#1e293b] rounded-lg p-3 flex flex-col justify-between hover:border-indigo-500/40 transition-colors">
              <div className="flex items-start justify-between">
                <div>
                  <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
                    Circadian Sleep
                  </span>
                  <div className="flex items-baseline gap-1.5 mt-0.5">
                    <span className="text-2xl font-bold text-indigo-300">
                      {twinState.sleepQuality.lastNightHours ?? 7.0}
                    </span>
                    <span className="text-[11px] text-slate-400">hrs</span>
                  </div>
                </div>

                <div className="w-8 h-8 rounded-full bg-indigo-500/10 border border-indigo-500/25 flex items-center justify-center text-indigo-400">
                  <Moon className="w-4 h-4" />
                </div>
              </div>

              <div className="mt-2 pt-2 border-t border-[#1e293b]/60 flex items-center justify-between text-[11px]">
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-400">Sleep Efficiency</span>
                  <span className="font-semibold text-slate-200">
                    {twinState.sleepQuality.efficiencyPercent ?? 82}%
                  </span>
                </div>
                <div className="text-right">
                  <span className="text-[10px] text-slate-400">Current Phase</span>
                  <span className="font-semibold text-indigo-400">
                    {twinState.sleepQuality.stage ?? 'Awake'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Organ Stress In Silico Biomarkers */}
          <div className="mt-4 pt-3.5 border-t border-[#1f293d]">
            <div className="flex items-center justify-between mb-2.5">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                  Organ-Level Physiological Strain Indexes
                </span>
              </div>
              <span className="text-[11px] text-slate-400">
                Calibrated against Virtual Patient Model
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {/* Pancreatic Beta-Cell Load */}
              <div className="bg-[#0b1120]/70 border border-[#1e293b] rounded-lg p-2.5">
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Pancreatic Beta-Cell Load</span>
                  <span
                    className={`font-bold ${
                      twinState.organStress.pancreasBetaCellLoad > 70
                        ? 'text-red-400'
                        : 'text-amber-400'
                    }`}
                  >
                    {twinState.organStress.pancreasBetaCellLoad}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${twinState.organStress.pancreasBetaCellLoad}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                    className={`h-full ${
                      twinState.organStress.pancreasBetaCellLoad > 70
                        ? 'bg-red-500'
                        : 'bg-amber-500'
                    }`}
                  />
                </div>
                <span className="text-[10px] text-slate-400 mt-1 block">
                  {twinState.organStress.pancreasBetaCellLoad > 70
                    ? 'Excessive insulin secretory demand'
                    : 'Compensatory basal secretion'}
                </span>
              </div>

              {/* Hepatic Glucose Output */}
              <div className="bg-[#0b1120]/70 border border-[#1e293b] rounded-lg p-2.5">
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Hepatic Glucose Production</span>
                  <span
                    className={`font-bold ${
                      twinState.organStress.hepaticGlucoseProduction > 70
                        ? 'text-red-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {twinState.organStress.hepaticGlucoseProduction}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${twinState.organStress.hepaticGlucoseProduction}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                    className={`h-full ${
                      twinState.organStress.hepaticGlucoseProduction > 70
                        ? 'bg-red-500'
                        : 'bg-emerald-500'
                    }`}
                  />
                </div>
                <span className="text-[10px] text-slate-400 mt-1 block">
                  {twinState.organStress.hepaticGlucoseProduction > 70
                    ? 'Incomplete postprandial hepatic suppression'
                    : 'Normal liver glycogen regulation'}
                </span>
              </div>

              {/* Cardiovascular Autonomic Strain */}
              <div className="bg-[#0b1120]/70 border border-[#1e293b] rounded-lg p-2.5">
                <div className="flex items-center justify-between text-xs mb-1.5">
                  <span className="text-slate-300 font-medium">Autonomic Strain</span>
                  <span
                    className={`font-bold ${
                      twinState.organStress.cardiovascularStrain > 60
                        ? 'text-amber-400'
                        : 'text-emerald-400'
                    }`}
                  >
                    {twinState.organStress.cardiovascularStrain}%
                  </span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${twinState.organStress.cardiovascularStrain}%` }}
                    transition={{ duration: 0.8, ease: 'easeOut' }}
                    className={`h-full ${
                      twinState.organStress.cardiovascularStrain > 60
                        ? 'bg-amber-500'
                        : 'bg-emerald-500'
                    }`}
                  />
                </div>
                <span className="text-[10px] text-slate-400 mt-1 block">
                  {twinState.organStress.cardiovascularStrain > 60
                    ? 'Sympathetic arousal & vascular tension'
                    : 'Balanced vagal nerve activity'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Footer info bar */}
        <div className="mt-4 pt-3 border-t border-[#1f293d] flex flex-col sm:flex-row items-center justify-between gap-2 text-[11px] text-slate-400">
          <div className="flex items-center gap-2">
            <span className="inline-block w-2 h-2 rounded-full bg-cyan-400" />
            <span>Digital Twin mathematical engine: Compartmental T2D Ordinary Differential Equations</span>
          </div>
          <button
            onClick={() => setIsModalOpen(true)}
            className="text-cyan-400 hover:text-cyan-300 font-medium flex items-center gap-1 transition-colors cursor-pointer"
          >
            <span>Simulate counter-measures</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </motion.div>

      {/* ========================================================================= */}
      {/* WHAT-IF SIMULATION MODAL                                                  */}
      {/* ========================================================================= */}
      <AnimatePresence>
        {isModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6 overflow-y-auto">
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsModalOpen(false)}
              className="fixed inset-0 bg-black/80 backdrop-blur-md transition-opacity"
            />

            {/* Modal Dialog */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 15 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 15 }}
              transition={{ duration: 0.25, ease: 'easeOut' }}
              className="relative w-full max-w-4xl bg-[#0f172a] border border-[#1e293b] rounded-2xl shadow-2xl p-5 sm:p-7 z-10 overflow-hidden my-auto"
            >
              {/* Modal Top Accent Glow */}
              <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 via-teal-400 to-indigo-500" />

              {/* Close Button */}
              <button
                onClick={() => setIsModalOpen(false)}
                className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors cursor-pointer"
                aria-label="Close Modal"
              >
                <X className="w-5 h-5" />
              </button>

              {/* Modal Header */}
              <div className="mb-6">
                <div className="flex items-center gap-2.5">
                  <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400">
                    <Sparkles className="w-4 h-4" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
                      In Silico 'What-If' Intervention Simulator
                    </h2>
                    <p className="text-xs text-slate-400">
                      Simulate dynamic physiological counter-measures for{' '}
                      <span className="text-cyan-300 font-semibold">{patientName}</span> to mitigate impending spike
                    </p>
                  </div>
                </div>
              </div>

              {/* Intervention Selection Tabs */}
              <div className="mb-6">
                <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2 block">
                  Select Digital Twin Clinical Counter-Measure:
                </label>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  {/* Option 1: Take Medication */}
                  <button
                    type="button"
                    onClick={() => setSelectedIntervention('take_medication')}
                    className={`text-left p-3.5 rounded-xl border transition-all cursor-pointer relative ${
                      selectedIntervention === 'take_medication'
                        ? 'bg-cyan-500/10 border-cyan-400/60 shadow-lg shadow-cyan-500/10 ring-1 ring-cyan-400/30'
                        : 'bg-[#111827] border-[#1e293b] hover:border-slate-700 text-slate-300'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`p-2 rounded-lg ${
                          selectedIntervention === 'take_medication'
                            ? 'bg-cyan-500/20 text-cyan-300'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        <Pill className="w-5 h-5" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-sm text-white">
                            Take Medication
                          </span>
                          <span className="text-xs font-bold text-cyan-400">-38 mg/dL</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                          Titrate Teneligliptin 20mg / Metformin to stimulate insulin release
                        </p>
                      </div>
                    </div>
                  </button>

                  {/* Option 2: Go for Walk */}
                  <button
                    type="button"
                    onClick={() => setSelectedIntervention('go_for_walk')}
                    className={`text-left p-3.5 rounded-xl border transition-all cursor-pointer relative ${
                      selectedIntervention === 'go_for_walk'
                        ? 'bg-emerald-500/10 border-emerald-400/60 shadow-lg shadow-emerald-500/10 ring-1 ring-emerald-400/30'
                        : 'bg-[#111827] border-[#1e293b] hover:border-slate-700 text-slate-300'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`p-2 rounded-lg ${
                          selectedIntervention === 'go_for_walk'
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        <Footprints className="w-5 h-5" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-sm text-white">
                            Go for Walk
                          </span>
                          <span className="text-xs font-bold text-emerald-400">-48 mg/dL</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                          20-min brisk walk triggering non-insulin GLUT4 muscle glucose uptake
                        </p>
                      </div>
                    </div>
                  </button>

                  {/* Option 3: Eat a Balanced Meal / Snack */}
                  <button
                    type="button"
                    onClick={() => setSelectedIntervention('eat_balanced_snack')}
                    className={`text-left p-3.5 rounded-xl border transition-all cursor-pointer relative ${
                      selectedIntervention === 'eat_balanced_snack'
                        ? 'bg-amber-500/10 border-amber-400/60 shadow-lg shadow-amber-500/10 ring-1 ring-amber-400/30'
                        : 'bg-[#111827] border-[#1e293b] hover:border-slate-700 text-slate-300'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div
                        className={`p-2 rounded-lg ${
                          selectedIntervention === 'eat_balanced_snack'
                            ? 'bg-amber-500/20 text-amber-300'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        <Utensils className="w-5 h-5" />
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <span className="font-semibold text-sm text-white">
                            Eat Fiber Snack
                          </span>
                          <span className="text-xs font-bold text-amber-400">-18 mg/dL</span>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-1 line-clamp-2">
                          High-fiber sprouts / roasted chana to slow down duodenal absorption
                        </p>
                      </div>
                    </div>
                  </button>
                </div>
              </div>

              {/* Simulation Trajectory Comparison Chart */}
              <div className="bg-[#0b1120] border border-[#1e293b] rounded-xl p-4 sm:p-5 mb-5 relative">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-3">
                  <div>
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                      <Zap className="w-4 h-4 text-cyan-400" />
                      Simulated 2-Hour Glycemic Trajectory (With vs Without Intervention)
                    </h4>
                    <p className="text-[11px] text-slate-400">
                      Comparison between projected unmitigated spike and digital twin counter-response
                    </p>
                  </div>

                  {/* Chart Legend */}
                  <div className="flex items-center gap-4 text-xs">
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-0.5 bg-red-400 border-b border-dashed border-red-400" />
                      <span className="text-slate-300 text-[11px]">Unmitigated Baseline</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-1 bg-cyan-400 rounded-full" />
                      <span className="text-cyan-300 font-semibold text-[11px]">Simulated Twin</span>
                    </div>
                  </div>
                </div>

                {/* Recharts Trajectory Graph */}
                <div className="h-64 w-full">
                  {isSimulating ? (
                    <div className="h-full flex flex-col items-center justify-center gap-2">
                      <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                      <span className="text-xs text-slate-400">
                        Running compartment ODE solver for {selectedIntervention}...
                      </span>
                    </div>
                  ) : (
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={trajectoryChartData}
                        margin={{ top: 10, right: 15, left: -20, bottom: 5 }}
                      >
                        <defs>
                          <linearGradient id="simulatedAreaGrad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.25} />
                            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                          </linearGradient>
                        </defs>

                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />

                        {/* Target Euglycemic Range (70 - 180) */}
                        <ReferenceArea
                          y1={70}
                          y2={180}
                          fill="#22c55e"
                          fillOpacity={0.05}
                          stroke="#22c55e"
                          strokeOpacity={0.12}
                        />

                        {/* Spike Threshold Reference */}
                        <ReferenceLine
                          y={180}
                          stroke="#f59e0b"
                          strokeDasharray="4 4"
                          strokeOpacity={0.7}
                          label={{
                            value: 'Spike Threshold (180)',
                            fill: '#f59e0b',
                            fontSize: 10,
                            position: 'insideTopRight'
                          }}
                        />

                        <XAxis
                          dataKey="time"
                          stroke="#64748b"
                          fontSize={11}
                          tickLine={false}
                        />
                        <YAxis
                          domain={[70, 280]}
                          stroke="#64748b"
                          fontSize={11}
                          tickLine={false}
                          unit=" mg/dL"
                        />

                        <Tooltip
                          content={({ active, payload, label }) => {
                            if (!active || !payload || !payload.length) return null;
                            const baseVal = payload.find((p) => p.dataKey === 'baseline')?.value;
                            const simVal = payload.find((p) => p.dataKey === 'simulated')?.value;
                            const diff =
                              typeof baseVal === 'number' && typeof simVal === 'number'
                                ? baseVal - simVal
                                : 0;
                            return (
                              <div className="bg-[#0b1120] border border-[#1e293b] p-3 rounded-lg shadow-xl text-xs space-y-1">
                                <p className="font-semibold text-slate-200">Time: +{label}</p>
                                <p className="text-red-400">
                                  Unmitigated: <strong className="text-white">{baseVal} mg/dL</strong>
                                </p>
                                <p className="text-cyan-400">
                                  With Intervention: <strong className="text-white">{simVal} mg/dL</strong>
                                </p>
                                {diff > 0 && (
                                  <p className="text-emerald-400 font-semibold pt-1 border-t border-[#1e293b]">
                                    Avoided Glycemic Surge: -{diff} mg/dL
                                  </p>
                                )}
                              </div>
                            );
                          }}
                        />

                        {/* Area under simulated line */}
                        <Area
                          type="monotone"
                          dataKey="simulated"
                          fill="url(#simulatedAreaGrad)"
                          stroke="transparent"
                        />

                        {/* Baseline Projected Unmitigated Spike (Dashed Red Line) */}
                        <Line
                          type="monotone"
                          dataKey="baseline"
                          name="Unmitigated Spike"
                          stroke="#ef4444"
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          dot={{ r: 3, fill: '#ef4444' }}
                        />

                        {/* Simulated Controlled Trajectory (Solid Cyan Line) */}
                        <Line
                          type="monotone"
                          dataKey="simulated"
                          name="With Counter-Measure"
                          stroke="#06b6d4"
                          strokeWidth={3}
                          dot={{ r: 4, fill: '#06b6d4', stroke: '#0f172a', strokeWidth: 2 }}
                          activeDot={{ r: 6, fill: '#06b6d4', stroke: '#ffffff', strokeWidth: 2 }}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  )}
                </div>

                {/* Key Simulation Impact Metrics */}
                {simulationData && (
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 pt-3 border-t border-[#1e293b]">
                    <div className="bg-[#111827] p-2.5 rounded-lg border border-[#1e293b]">
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
                        Predicted Delta
                      </span>
                      <span className="text-lg font-bold text-emerald-400">
                        {simulationData.predictedDeltaMgDl} mg/dL
                      </span>
                    </div>

                    <div className="bg-[#111827] p-2.5 rounded-lg border border-[#1e293b]">
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
                        Time to Stabilize
                      </span>
                      <span className="text-lg font-bold text-cyan-300">
                        ~{simulationData.estimatedTimeToStabilizeMin} mins
                      </span>
                    </div>

                    <div className="bg-[#111827] p-2.5 rounded-lg border border-[#1e293b]">
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
                        Spike Outcome
                      </span>
                      <span className="text-xs font-bold text-emerald-400 flex items-center gap-1 mt-1">
                        <ShieldCheck className="w-4 h-4 text-emerald-400" />
                        Spike Prevented
                      </span>
                    </div>

                    <div className="bg-[#111827] p-2.5 rounded-lg border border-[#1e293b]">
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider block">
                        Target Status
                      </span>
                      <span className="text-xs font-bold text-slate-200 mt-1 block">
                        Euglycemia Maintained
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Clinical Mechanism & Physiological Rationale */}
              {simulationData && (
                <div className="bg-cyan-950/20 border border-cyan-500/30 rounded-xl p-3.5 mb-6 text-xs flex items-start gap-3">
                  <div className="p-1.5 rounded-md bg-cyan-500/20 text-cyan-400 shrink-0 mt-0.5">
                    <Info className="w-4 h-4" />
                  </div>
                  <div>
                    <h5 className="font-semibold text-cyan-200 mb-0.5">
                      Digital Twin Physiological Mechanism
                    </h5>
                    <p className="text-slate-300 leading-relaxed text-[11px]">
                      {simulationData.clinicalRationale}
                    </p>
                  </div>
                </div>
              )}

              {/* Modal Footer Controls */}
              <div className="flex flex-col-reverse sm:flex-row items-center justify-between gap-3 pt-3 border-t border-[#1e293b]">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="w-full sm:w-auto px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-medium transition-colors cursor-pointer"
                >
                  Close Simulator
                </button>

                <div className="flex items-center gap-3 w-full sm:w-auto">
                  <motion.button
                    whileTap={{ scale: 0.98 }}
                    onClick={() => {
                      setIsApplied(true);
                      setTimeout(() => {
                        setIsApplied(false);
                        setIsModalOpen(false);
                      }, 1200);
                    }}
                    className={`w-full sm:w-auto px-5 py-2.5 rounded-lg font-semibold text-xs flex items-center justify-center gap-2 transition-all cursor-pointer ${
                      isApplied
                        ? 'bg-emerald-500 text-slate-950'
                        : 'bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-slate-950 shadow-lg shadow-cyan-500/25'
                    }`}
                  >
                    {isApplied ? (
                      <>
                        <Check className="w-4 h-4" />
                        <span>Recommended to Patient Plan</span>
                      </>
                    ) : (
                      <>
                        <ShieldCheck className="w-4 h-4" />
                        <span>Deploy Recommendation to Care Plan</span>
                      </>
                    )}
                  </motion.button>
                </div>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </>
  );
}
