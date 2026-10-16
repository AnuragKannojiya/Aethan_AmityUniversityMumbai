/**
 * GlucoTwin - Clinical Utility Functions
 * Amity University Mumbai - Team Aethan
 */

import { RiskLevel, GlucoseTrend } from './types';

/**
 * Returns color hex code according to glucose range:
 * - Red (>250 mg/dL): Severe Hyperglycemia
 * - Orange (180-250 mg/dL): Hyperglycemia
 * - Green (70-180 mg/dL): Target Euglycemic Range (Time-in-Range)
 * - Yellow (<70 mg/dL): Hypoglycemia
 */
export function getGlucoseColor(glucose: number): string {
  if (glucose > 250) return '#ef4444'; // Red
  if (glucose >= 180) return '#f97316'; // Orange
  if (glucose >= 70) return '#22c55e'; // Green
  return '#eab308'; // Yellow
}

export function getGlucoseBadgeClass(glucose: number): { bg: string; text: string; border: string; label: string } {
  if (glucose > 250) {
    return {
      bg: 'bg-red-950/40',
      text: 'text-red-400',
      border: 'border-red-500/50',
      label: 'Severe Hyperglycemia'
    };
  }
  if (glucose >= 180) {
    return {
      bg: 'bg-amber-950/40',
      text: 'text-amber-400',
      border: 'border-amber-500/50',
      label: 'Elevated (Above Target)'
    };
  }
  if (glucose >= 70) {
    return {
      bg: 'bg-emerald-950/40',
      text: 'text-emerald-400',
      border: 'border-emerald-500/50',
      label: 'In Target Range'
    };
  }
  return {
    bg: 'bg-yellow-950/40',
    text: 'text-yellow-400',
    border: 'border-yellow-500/50',
    label: 'Hypoglycemia Alert'
  };
}

export function getRiskColor(risk: RiskLevel): string {
  switch (risk) {
    case 'Critical': return '#ef4444'; // Red
    case 'High': return '#f97316'; // Orange
    case 'Moderate': return '#f59e0b'; // Amber / Yellow
    case 'Low': return '#22c55e'; // Green
    default: return '#06b6d4';
  }
}

export function getRiskBadgeClasses(risk: RiskLevel): string {
  switch (risk) {
    case 'Critical':
      return 'bg-red-500/15 text-red-400 border border-red-500/30';
    case 'High':
      return 'bg-orange-500/15 text-orange-400 border border-orange-500/30';
    case 'Moderate':
      return 'bg-amber-500/15 text-amber-400 border border-amber-500/30';
    case 'Low':
      return 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30';
    default:
      return 'bg-slate-500/15 text-slate-400 border border-slate-500/30';
  }
}

export function getTrendIcon(trend: GlucoseTrend): { symbol: string; text: string; color: string } {
  switch (trend) {
    case 'rising_fast':
      return { symbol: '↑↑', text: 'Rising fast (>2 mg/dL/min)', color: 'text-red-400' };
    case 'rising':
      return { symbol: '↗', text: 'Rising (1-2 mg/dL/min)', color: 'text-amber-400' };
    case 'stable':
      return { symbol: '→', text: 'Stable (±1 mg/dL/min)', color: 'text-emerald-400' };
    case 'falling':
      return { symbol: '↘', text: 'Falling (1-2 mg/dL/min)', color: 'text-cyan-400' };
    case 'falling_fast':
      return { symbol: '↓↓', text: 'Falling fast (>2 mg/dL/min)', color: 'text-yellow-400' };
  }
}
