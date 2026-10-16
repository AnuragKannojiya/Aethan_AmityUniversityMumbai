'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Bell,
  AlertOctagon,
  AlertTriangle,
  AlertCircle,
  Info,
  ChevronDown,
  ChevronUp,
  Stethoscope,
  CheckCircle,
  Clock,
  ShieldCheck
} from 'lucide-react';
import { PatientAlert, AlertSeverity } from '../lib/types';

interface AlertPanelProps {
  alerts: PatientAlert[];
}

export default function AlertPanel({ alerts: initialAlerts }: AlertPanelProps) {
  const [alerts, setAlerts] = useState<PatientAlert[]>(initialAlerts);
  const [expandedAlertId, setExpandedAlertId] = useState<string | null>(
    initialAlerts.length > 0 ? initialAlerts[0].id : null
  );
  const [filterSeverity, setFilterSeverity] = useState<'all' | 'critical' | 'active'>('all');

  // Severity ordering weight
  const severityWeight: Record<AlertSeverity, number> = {
    critical: 4,
    high: 3,
    moderate: 2,
    low: 1,
  };

  // Sort by severity (critical first) and unacknowledged status
  const sortedAlerts = [...alerts].sort((a, b) => {
    if (a.isAcknowledged !== b.isAcknowledged) {
      return a.isAcknowledged ? 1 : -1;
    }
    return severityWeight[b.severity] - severityWeight[a.severity];
  });

  const filteredAlerts = sortedAlerts.filter((item) => {
    if (filterSeverity === 'critical') return item.severity === 'critical';
    if (filterSeverity === 'active') return !item.isAcknowledged;
    return true;
  });

  const toggleExpand = (id: string) => {
    setExpandedAlertId(expandedAlertId === id ? null : id);
  };

  const handleAcknowledge = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setAlerts((prev) =>
      prev.map((a) => (a.id === id ? { ...a, isAcknowledged: !a.isAcknowledged } : a))
    );
  };

  const getSeverityConfig = (severity: AlertSeverity) => {
    switch (severity) {
      case 'critical':
        return {
          icon: <AlertOctagon className="w-4 h-4 text-red-400" />,
          badgeClass: 'bg-red-950/80 text-red-300 border-red-700/60',
          borderClass: 'border-red-900/50 hover:border-red-700',
          bgClass: 'bg-red-950/15',
          pulse: true,
          label: 'CRITICAL',
        };
      case 'high':
        return {
          icon: <AlertTriangle className="w-4 h-4 text-orange-400" />,
          badgeClass: 'bg-orange-950/80 text-orange-300 border-orange-700/60',
          borderClass: 'border-orange-900/50 hover:border-orange-700',
          bgClass: 'bg-orange-950/10',
          pulse: false,
          label: 'HIGH RISK',
        };
      case 'moderate':
        return {
          icon: <AlertCircle className="w-4 h-4 text-amber-400" />,
          badgeClass: 'bg-amber-950/80 text-amber-300 border-amber-700/60',
          borderClass: 'border-amber-900/50 hover:border-amber-700',
          bgClass: 'bg-amber-950/10',
          pulse: false,
          label: 'MODERATE',
        };
      case 'low':
        return {
          icon: <Info className="w-4 h-4 text-cyan-400" />,
          badgeClass: 'bg-cyan-950/80 text-cyan-300 border-cyan-700/60',
          borderClass: 'border-cyan-900/50 hover:border-cyan-700',
          bgClass: 'bg-cyan-950/10',
          pulse: false,
          label: 'ROUTINE',
        };
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="bg-[#111827] border border-[#1f293d] rounded-xl p-5 shadow-lg flex flex-col justify-between h-full"
      aria-label="Clinical Decision Support Alert Panel"
    >
      <div>
        {/* Top Header & Filter Pills */}
        <div className="flex items-center justify-between pb-3 border-b border-[#1f293d]">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Bell className="w-5 h-5 text-cyan-400" />
              {alerts.some((a) => a.severity === 'critical' && !a.isAcknowledged) && (
                <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-red-500 animate-ping" />
              )}
            </div>
            <h2 className="text-base font-semibold text-white tracking-tight">
              Clinical Alert Feed
            </h2>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
              {filteredAlerts.length}
            </span>
          </div>

          {/* Severity Filter Tabs */}
          <div className="flex items-center gap-1 text-[11px]">
            <button
              onClick={() => setFilterSeverity('all')}
              className={`px-2 py-1 rounded transition-colors ${
                filterSeverity === 'all'
                  ? 'bg-slate-700 text-white font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilterSeverity('critical')}
              className={`px-2 py-1 rounded transition-colors ${
                filterSeverity === 'critical'
                  ? 'bg-red-950 text-red-300 border border-red-800/50 font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Critical
            </button>
            <button
              onClick={() => setFilterSeverity('active')}
              className={`px-2 py-1 rounded transition-colors ${
                filterSeverity === 'active'
                  ? 'bg-cyan-950 text-cyan-300 border border-cyan-800/50 font-medium'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Active
            </button>
          </div>
        </div>

        {/* Alerts List */}
        <div className="mt-3 space-y-2.5 max-h-[340px] overflow-y-auto pr-1">
          {filteredAlerts.length === 0 ? (
            <div className="text-center py-8 text-slate-500 text-xs flex flex-col items-center">
              <ShieldCheck className="w-8 h-8 text-emerald-400/60 mb-2" />
              <span>No alerts matching the selected filter</span>
            </div>
          ) : (
            <AnimatePresence>
              {filteredAlerts.map((alert) => {
                const config = getSeverityConfig(alert.severity);
                const isExpanded = expandedAlertId === alert.id;

                return (
                  <motion.div
                    key={alert.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: 10 }}
                    transition={{ duration: 0.25 }}
                    onClick={() => toggleExpand(alert.id)}
                    className={`rounded-lg border p-3 cursor-pointer transition-all ${
                      config.borderClass
                    } ${config.bgClass} ${
                      alert.isAcknowledged ? 'opacity-60 saturate-50' : ''
                    } ${config.pulse && !alert.isAcknowledged ? 'animate-critical-glow' : ''}`}
                  >
                    {/* Alert Card Header */}
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-start gap-2.5">
                        <div className="mt-0.5 shrink-0">{config.icon}</div>
                        <div>
                          <div className="flex items-center gap-2 flex-wrap">
                            <span
                              className={`text-[10px] font-bold px-1.5 py-0.2 rounded border font-mono ${config.badgeClass}`}
                            >
                              {config.label}
                            </span>
                            <span className="text-xs font-semibold text-slate-200">
                              {alert.title}
                            </span>
                          </div>
                          <p className="text-xs text-slate-300 mt-1 leading-snug">
                            {alert.message}
                          </p>
                        </div>
                      </div>

                      <div className="flex flex-col items-end shrink-0 gap-1">
                        <span className="text-[10px] text-slate-400 flex items-center gap-1 font-mono">
                          <Clock className="w-3 h-3 text-slate-500" />
                          {alert.displayTime}
                        </span>
                        <div className="text-slate-400 mt-1">
                          {isExpanded ? (
                            <ChevronUp className="w-4 h-4 text-cyan-400" />
                          ) : (
                            <ChevronDown className="w-4 h-4 text-slate-500" />
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Expandable Recommended Action */}
                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          transition={{ duration: 0.2 }}
                          className="mt-3 pt-3 border-t border-slate-700/60 overflow-hidden"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <div className="bg-[#0a0e1a]/90 p-2.5 rounded border border-cyan-900/40 text-xs">
                            <div className="flex items-center gap-1.5 text-cyan-300 font-semibold mb-1">
                              <Stethoscope className="w-3.5 h-3.5" />
                              <span>Clinical Recommendation & Intervention:</span>
                            </div>
                            <p className="text-slate-300 leading-relaxed text-[11px]">
                              {alert.recommendedAction}
                            </p>
                          </div>

                          <div className="mt-2.5 flex items-center justify-between">
                            <span className="text-[10px] text-slate-500">
                              Alert ID: {alert.id}
                            </span>
                            <button
                              onClick={(e) => handleAcknowledge(alert.id, e)}
                              className={`text-xs px-2.5 py-1 rounded font-medium flex items-center gap-1.5 transition-colors ${
                                alert.isAcknowledged
                                  ? 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                                  : 'bg-cyan-600 hover:bg-cyan-500 text-white shadow-sm'
                              }`}
                            >
                              <CheckCircle className="w-3.5 h-3.5" />
                              {alert.isAcknowledged ? 'Mark as Unacknowledged' : 'Acknowledge Alert'}
                            </button>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })}
            </AnimatePresence>
          )}
        </div>
      </div>

      {/* Footer Info */}
      <div className="pt-3 border-t border-[#1f293d] mt-3 flex items-center justify-between text-[11px] text-slate-500">
        <span>Clinical Decision Support Engine v2.4</span>
        <span>Auto-escalates critical alerts to SMS / Hospital EMR</span>
      </div>
    </motion.section>
  );
}
