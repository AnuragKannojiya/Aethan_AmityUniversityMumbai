'use client';

/**
 * GlucoTwin - Clinical Digital Twin Glucose Spike Prediction Dashboard
 * Multi-Modal Physiological Fusion (Static EHR + Dynamic CGM Telemetry)
 * Team Aethan - Amity University Mumbai
 */

import React, { useState, useEffect, useCallback, useTransition } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Heart,
  RefreshCw,
  Clock,
  ShieldAlert,
  ChevronDown,
  User,
  Radio,
  Sparkles,
  Sliders,
  HelpCircle,
  ExternalLink,
  Layers,
  CheckCircle2,
  AlertTriangle
} from 'lucide-react';

import { Patient, DashboardData } from '../lib/types';
import { getPatients, getDashboard } from '../lib/api';
import { MOCK_PATIENTS, getMockDashboardData } from '../lib/mockData';
import { getRiskBadgeClasses } from '../lib/utils';

// Import the 6 Dashboard Components
import PatientOverview from '../components/PatientOverview';
import GlucoseChart from '../components/GlucoseChart';
import RiskGauge from '../components/RiskGauge';
import PredictionTimeline from '../components/PredictionTimeline';
import AlertPanel from '../components/AlertPanel';
import TwinVisualization from '../components/TwinVisualization';

export default function DashboardPage() {
  const [patients, setPatients] = useState<Patient[]>(MOCK_PATIENTS);
  const [selectedPatientId, setSelectedPatientId] = useState<string>(MOCK_PATIENTS[0].id);
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<Date | null>(null);
  const [isDropdownOpen, setIsDropdownOpen] = useState<boolean>(false);
  const [isPending, startTransition] = useTransition();

  // Fetch initial patient cohort list
  useEffect(() => {
    let mounted = true;
    async function loadPatients() {
      try {
        const cohort = await getPatients();
        if (mounted && cohort && cohort.length > 0) {
          setPatients(cohort);
        }
      } catch (err) {
        console.warn('Using fallback mock patients cohort', err);
      }
    }
    loadPatients();
    return () => {
      mounted = false;
    };
  }, []);

  // Fetch consolidated dashboard data for selected patient
  const loadDashboard = useCallback(
    async (patientId: string, showSpinner = false) => {
      if (showSpinner) setIsRefreshing(true);
      try {
        const data = await getDashboard(patientId);
        setDashboardData(data);
        setLastRefreshedAt(new Date());
      } catch (err) {
        console.warn('Dashboard fetch fallback to mock', err);
        setDashboardData(getMockDashboardData(patientId));
        setLastRefreshedAt(new Date());
      } finally {
        setIsLoading(false);
        if (showSpinner) setIsRefreshing(false);
      }
    },
    []
  );

  // Trigger load on selected patient change
  useEffect(() => {
    setIsLoading(true);
    loadDashboard(selectedPatientId, false);
  }, [selectedPatientId, loadDashboard]);

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      loadDashboard(selectedPatientId, true);
    }, 30000);

    return () => clearInterval(interval);
  }, [selectedPatientId, loadDashboard]);

  const handlePatientSelect = (id: string) => {
    setSelectedPatientId(id);
    setIsDropdownOpen(false);
  };

  const selectedPatient =
    patients.find((p) => p.id === selectedPatientId) ||
    dashboardData?.patient ||
    patients[0];

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-slate-100 flex flex-col font-sans selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* ===================================================================== */}
      {/* TOP HEADER / APP BAR                                                 */}
      {/* ===================================================================== */}
      <header className="sticky top-0 z-40 bg-[#0c1222]/90 backdrop-blur-md border-b border-[#1f293d] shadow-md px-4 sm:px-6 lg:px-8 py-3.5">
        <div className="max-w-[1680px] mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          {/* Logo & Platform Info */}
          <div className="flex items-center gap-3.5 w-full md:w-auto justify-between md:justify-start">
            <div className="flex items-center gap-3">
              {/* Animated Icon Badge */}
              <div className="relative flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 shadow-lg shadow-cyan-500/20 text-slate-950">
                <Activity className="w-5 h-5 text-white animate-pulse" />
                <span className="absolute -top-1 -right-1 flex h-3 w-3">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500" />
                </span>
              </div>

              <div>
                <div className="flex items-center gap-2">
                  <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-1.5">
                    Gluco<span className="text-cyan-400">Twin</span>
                  </h1>
                  <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-cyan-950/60 border border-cyan-500/40 text-cyan-300">
                    v2.6 In-Silico
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 font-medium">
                  Continuous T2D Spike Prediction Digital Twin • <span className="text-slate-300">Team Aethan</span>
                </p>
              </div>
            </div>

            {/* Mobile Refresh Button */}
            <div className="flex md:hidden items-center gap-2">
              <button
                onClick={() => loadDashboard(selectedPatientId, true)}
                disabled={isRefreshing}
                className="p-2 rounded-lg bg-slate-800 text-slate-300 hover:text-white border border-slate-700 active:scale-95 transition-all"
                aria-label="Refresh telemetry"
              >
                <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin text-cyan-400' : ''}`} />
              </button>
            </div>
          </div>

          {/* Right Header Controls: Patient Selector & Telemetry Status */}
          <div className="flex flex-wrap items-center justify-end gap-3 w-full md:w-auto">
            {/* Live Telemetry Heartbeat Status */}
            <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#111827] border border-[#1f293d] text-xs">
              <Radio className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
              <span className="text-slate-300 text-[11px] font-medium">Wearable CGM Live</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
            </div>

            {/* Patient Selector Dropdown */}
            <div className="relative">
              <button
                type="button"
                onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                className="flex items-center gap-2.5 px-3.5 py-1.5 rounded-lg bg-[#111827] border border-[#1f293d] hover:border-cyan-500/50 text-left transition-all shadow-sm cursor-pointer"
              >
                <div className="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-300 flex items-center justify-center font-bold text-xs border border-cyan-500/30">
                  {selectedPatient.name.charAt(0)}
                </div>
                <div>
                  <div className="flex items-center gap-1.5">
                    <span className="text-xs font-semibold text-white">
                      {selectedPatient.name}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.2 rounded ${getRiskBadgeClasses(
                        selectedPatient.currentRisk
                      )}`}
                    >
                      {selectedPatient.currentRisk}
                    </span>
                  </div>
                  <span className="text-[10px] text-slate-400">
                    {selectedPatient.id} • {selectedPatient.city.split(',')[0]}
                  </span>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-400 ml-1 transition-transform ${isDropdownOpen ? 'rotate-180' : ''}`} />
              </button>

              {/* Dropdown Menu */}
              <AnimatePresence>
                {isDropdownOpen && (
                  <>
                    <div
                      className="fixed inset-0 z-40"
                      onClick={() => setIsDropdownOpen(false)}
                    />
                    <motion.div
                      initial={{ opacity: 0, y: -8 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -8 }}
                      transition={{ duration: 0.15 }}
                      className="absolute right-0 mt-2 w-72 bg-[#111827] border border-[#1f293d] rounded-xl shadow-2xl py-2 z-50 overflow-hidden"
                    >
                      <div className="px-3 py-1.5 border-b border-[#1f293d] flex items-center justify-between">
                        <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
                          Monitored T2D Cohort ({patients.length})
                        </span>
                        <span className="text-[10px] text-cyan-400 font-medium">India Cohort</span>
                      </div>

                      <div className="max-h-72 overflow-y-auto divide-y divide-[#1f293d]/50">
                        {patients.map((p) => {
                          const isSelected = p.id === selectedPatientId;
                          return (
                            <button
                              key={p.id}
                              type="button"
                              onClick={() => handlePatientSelect(p.id)}
                              className={`w-full px-3.5 py-2.5 text-left flex items-center justify-between hover:bg-[#1a2333] transition-colors cursor-pointer ${
                                isSelected ? 'bg-cyan-500/10' : ''
                              }`}
                            >
                              <div className="flex items-center gap-2.5">
                                <div
                                  className={`w-7 h-7 rounded-full flex items-center justify-center font-bold text-xs ${
                                    isSelected
                                      ? 'bg-cyan-500 text-slate-950'
                                      : 'bg-slate-800 text-slate-300'
                                  }`}
                                >
                                  {p.name.charAt(0)}
                                </div>
                                <div>
                                  <div className="text-xs font-semibold text-white flex items-center gap-1.5">
                                    {p.name}
                                    {isSelected && (
                                      <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400" />
                                    )}
                                  </div>
                                  <div className="text-[11px] text-slate-400">
                                    {p.id} • {p.age}y • HbA1c {p.hba1c}%
                                  </div>
                                </div>
                              </div>

                              <span
                                className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${getRiskBadgeClasses(
                                  p.currentRisk
                                )}`}
                              >
                                {p.currentRisk}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </motion.div>
                  </>
                )}
              </AnimatePresence>
            </div>

            {/* Last Updated & Manual Refresh Trigger */}
            <div className="flex items-center gap-2 pl-2 border-l border-[#1f293d]">
              <div className="hidden lg:flex flex-col text-right">
                <span className="text-[10px] text-slate-400 uppercase tracking-wider flex items-center gap-1 justify-end">
                  <Clock className="w-3 h-3 text-cyan-400" />
                  Telemetry Sync
                </span>
                <span className="text-[11px] font-medium text-slate-200">
                  {lastRefreshedAt
                    ? lastRefreshedAt.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                        second: '2-digit',
                      })
                    : 'Just now'}
                </span>
              </div>

              <motion.button
                whileTap={{ scale: 0.95 }}
                onClick={() => loadDashboard(selectedPatientId, true)}
                disabled={isRefreshing}
                title="Refresh digital twin telemetry (30s auto-refresh active)"
                className="hidden md:flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#111827] hover:bg-[#1a2333] border border-[#1f293d] text-slate-300 hover:text-white text-xs font-medium transition-all shadow-sm cursor-pointer disabled:opacity-50"
              >
                <RefreshCw
                  className={`w-3.5 h-3.5 text-cyan-400 ${
                    isRefreshing ? 'animate-spin' : ''
                  }`}
                />
                <span className="text-[11px]">{isRefreshing ? 'Syncing...' : 'Refresh'}</span>
              </motion.button>
            </div>
          </div>
        </div>
      </header>

      {/* ===================================================================== */}
      {/* MAIN DASHBOARD CONTAINER                                              */}
      {/* ===================================================================== */}
      <main className="flex-1 max-w-[1680px] w-full mx-auto p-4 sm:p-6 lg:p-8 space-y-6">
        {isLoading && !dashboardData ? (
          /* ===================================================================== */
          /* SKELETON LOADING VIEW                                                 */
          /* ===================================================================== */
          <div className="space-y-6 animate-pulse">
            {/* Top Row Skeleton: 1 col + 2 cols + 1 col */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
              <div className="h-96 bg-[#111827] rounded-xl border border-[#1f293d] p-5 space-y-4">
                <div className="h-6 w-3/4 bg-slate-800 rounded" />
                <div className="h-20 w-20 rounded-full bg-slate-800 mx-auto" />
                <div className="h-4 w-1/2 bg-slate-800 mx-auto rounded" />
                <div className="space-y-2 pt-4">
                  <div className="h-3 bg-slate-800 rounded" />
                  <div className="h-3 bg-slate-800 rounded" />
                  <div className="h-3 bg-slate-800 rounded" />
                </div>
              </div>

              <div className="lg:col-span-2 h-96 bg-[#111827] rounded-xl border border-[#1f293d] p-5 flex flex-col justify-between">
                <div className="flex justify-between">
                  <div className="h-6 w-1/3 bg-slate-800 rounded" />
                  <div className="h-6 w-24 bg-slate-800 rounded" />
                </div>
                <div className="h-56 bg-slate-800/40 rounded-lg flex items-center justify-center">
                  <span className="text-xs text-slate-500 font-medium">Loading multimodal telemetry...</span>
                </div>
              </div>

              <div className="h-96 bg-[#111827] rounded-xl border border-[#1f293d] p-5 flex flex-col justify-between">
                <div className="h-6 w-1/2 bg-slate-800 rounded" />
                <div className="w-36 h-36 rounded-full border-8 border-slate-800 mx-auto" />
                <div className="h-10 bg-slate-800 rounded" />
              </div>
            </div>

            {/* Middle Row Skeleton: Prediction Timeline (full width) */}
            <div className="h-80 bg-[#111827] rounded-xl border border-[#1f293d] p-5 space-y-4">
              <div className="flex justify-between">
                <div className="h-6 w-1/4 bg-slate-800 rounded" />
                <div className="h-6 w-32 bg-slate-800 rounded" />
              </div>
              <div className="h-44 bg-slate-800/40 rounded-lg" />
              <div className="grid grid-cols-3 gap-4 pt-2">
                <div className="h-12 bg-slate-800 rounded" />
                <div className="h-12 bg-slate-800 rounded" />
                <div className="h-12 bg-slate-800 rounded" />
              </div>
            </div>

            {/* Bottom Row Skeleton: 2 cols */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="h-96 bg-[#111827] rounded-xl border border-[#1f293d] p-5 space-y-3">
                <div className="h-6 w-1/3 bg-slate-800 rounded" />
                <div className="h-20 bg-slate-800/60 rounded" />
                <div className="h-20 bg-slate-800/60 rounded" />
                <div className="h-20 bg-slate-800/60 rounded" />
              </div>
              <div className="h-96 bg-[#111827] rounded-xl border border-[#1f293d] p-5 space-y-3">
                <div className="h-6 w-1/3 bg-slate-800 rounded" />
                <div className="grid grid-cols-4 gap-2">
                  <div className="h-20 bg-slate-800 rounded" />
                  <div className="h-20 bg-slate-800 rounded" />
                  <div className="h-20 bg-slate-800 rounded" />
                  <div className="h-20 bg-slate-800 rounded" />
                </div>
                <div className="h-32 bg-slate-800/40 rounded" />
              </div>
            </div>
          </div>
        ) : dashboardData ? (
          /* ===================================================================== */
          /* COMPLETE 6-SECTION CLINICAL DASHBOARD GRID                             */
          /* ===================================================================== */
          <div className="space-y-6">
            {/* ----------------------------------------------------------------- */}
            {/* ROW 1: PatientOverview (1 col) | GlucoseChart (2 cols) | RiskGauge (1 col) */}
            {/* ----------------------------------------------------------------- */}
            <section className="grid grid-cols-1 lg:grid-cols-4 gap-6 items-stretch">
              {/* Section 1: PatientOverview (Top-Left, 1 col) */}
              <div className="lg:col-span-1 flex flex-col">
                <PatientOverview patient={dashboardData.patient} />
              </div>

              {/* Section 2: GlucoseChart (Top-Center, Spans 2 cols) */}
              <div className="lg:col-span-2 flex flex-col">
                <GlucoseChart
                  historicalReadings={dashboardData.historicalReadings}
                  prediction={dashboardData.prediction}
                  patientName={dashboardData.patient.name}
                />
              </div>

              {/* Section 3: RiskGauge (Top-Right, 1 col) */}
              <div className="lg:col-span-1 flex flex-col">
                <RiskGauge
                  riskScore={dashboardData.patient.currentRiskScore}
                  riskLevel={dashboardData.patient.currentRisk}
                  timeToSpikeMin={dashboardData.patient.timeToPotentialSpikeMin}
                  predictedPeakGlucose={dashboardData.prediction.predictedPeakGlucose}
                />
              </div>
            </section>

            {/* ----------------------------------------------------------------- */}
            {/* ROW 2: PredictionTimeline (Middle, Full Width)                     */}
            {/* ----------------------------------------------------------------- */}
            <section className="w-full">
              <PredictionTimeline
                prediction={dashboardData.prediction}
                currentGlucose={dashboardData.patient.lastGlucose}
              />
            </section>

            {/* ----------------------------------------------------------------- */}
            {/* ROW 3: AlertPanel (Bottom-Left) | TwinVisualization (Bottom-Right) */}
            {/* ----------------------------------------------------------------- */}
            <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
              {/* Section 5: AlertPanel (Bottom-Left) */}
              <div className="flex flex-col">
                <AlertPanel
                  key={`alerts-${dashboardData.patient.id}`}
                  alerts={dashboardData.alerts}
                />
              </div>

              {/* Section 6: TwinVisualization (Bottom-Right) */}
              <div className="flex flex-col">
                <TwinVisualization
                  key={`twin-${dashboardData.patient.id}`}
                  twinState={dashboardData.twinState}
                  patientId={dashboardData.patient.id}
                  patientName={dashboardData.patient.name}
                  historicalReadings={dashboardData.historicalReadings}
                />
              </div>
            </section>
          </div>
        ) : (
          /* Error State Fallback */
          <div className="text-center py-20 bg-[#111827] rounded-xl border border-[#1f293d]">
            <AlertTriangle className="w-12 h-12 text-amber-400 mx-auto mb-3" />
            <h3 className="text-lg font-bold text-white">No Telemetry Stream Detected</h3>
            <p className="text-sm text-slate-400 mt-1 mb-4">
              Unable to reach GlucoTwin edge telemetry server.
            </p>
            <button
              onClick={() => loadDashboard(selectedPatientId, true)}
              className="px-4 py-2 bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-xs rounded-lg transition-colors cursor-pointer"
            >
              Retry Connection
            </button>
          </div>
        )}
      </main>

      {/* ===================================================================== */}
      {/* FOOTER                                                                */}
      {/* ===================================================================== */}
      <footer className="border-t border-[#1f293d] bg-[#0c1222] py-4 px-6 text-xs text-slate-400 mt-12">
        <div className="max-w-[1680px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-slate-300">GlucoTwin Digital Twin Platform</span>
            <span>•</span>
            <span>Digital Twin Challenge 2026</span>
            <span>•</span>
            <span className="text-cyan-400 font-medium">Team Aethan (Amity University Mumbai)</span>
          </div>

          <div className="flex items-center gap-4 text-slate-400">
            <span>Synthetic Cohort Data (Synthea EHR + Simulated CGM)</span>
            <span>•</span>
            <span className="text-emerald-400 flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              Bi-directional In Silico Sync
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
