'use client';

import React from 'react';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  CartesianGrid
} from 'recharts';
import {
  Clock,
  Sparkles,
  AlertTriangle,
  Flame,
  Layers,
  ArrowRight
} from 'lucide-react';
import { SpikePrediction } from '../lib/types';
import { getGlucoseColor } from '../lib/utils';

interface PredictionTimelineProps {
  prediction: SpikePrediction;
  currentGlucose: number;
}

export default function PredictionTimeline({
  prediction,
  currentGlucose,
}: PredictionTimelineProps) {
  // Build timeline dataset starting from Now (+0m) to +120m
  const timelineData = [
    {
      time: 'Now',
      minutes: 0,
      predictedGlucose: currentGlucose,
      confidenceLower: currentGlucose,
      confidenceUpper: currentGlucose,
      confidenceSpread: 0,
      isPeak: false,
    },
    ...prediction.projectedReadings.map((pt, idx) => {
      const minutes = (idx + 1) * 15;
      const isPeak = pt.predictedGlucose === prediction.predictedPeakGlucose;
      return {
        time: pt.time,
        minutes,
        predictedGlucose: pt.predictedGlucose,
        confidenceLower: pt.confidenceLower,
        confidenceUpper: pt.confidenceUpper,
        confidenceSpread: pt.confidenceUpper - pt.confidenceLower,
        isPeak,
      };
    }),
  ];

  return (
    <motion.section
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.15 }}
      className="bg-[#111827] border border-[#1f293d] rounded-xl p-5 shadow-lg flex flex-col justify-between"
      aria-label="2-Hour Prediction Trajectory and Risk Factor Attribution"
    >
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-cyan-400" />
              2-Hour Physiological Trajectory & Confidence Bounds
            </h2>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 font-mono">
              Horizon: +120 min
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Forward simulation of active digestive carbohydrate absorption, metabolic clearance, and physical activity
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-[11px] text-slate-300">
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-1 bg-cyan-400 rounded-full" />
            <span className="font-medium text-cyan-300">Mean Trajectory</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-2.5 bg-cyan-500/20 border border-cyan-500/40 rounded-sm" />
            <span>95% Confidence Band</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3.5 h-0.5 border-b-2 border-dashed border-amber-500" />
            <span className="text-amber-400">Spike Threshold (180 mg/dL)</span>
          </div>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="w-full h-[220px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={timelineData}
            margin={{ top: 15, right: 20, left: -10, bottom: 0 }}
          >
            <defs>
              <linearGradient id="predictionAreaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.35} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.02} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" vertical={false} />

            <XAxis
              dataKey="time"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
            />

            <YAxis
              domain={[60, 320]}
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              unit=" mg/dL"
              ticks={[70, 120, 180, 240, 300]}
            />

            <Tooltip content={<CustomPredictionTooltip />} />

            {/* Spike Threshold Reference */}
            <ReferenceLine
              y={180}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeWidth={1.5}
              label={{
                value: 'Hyperglycemia Cutoff (180 mg/dL)',
                fill: '#f59e0b',
                fontSize: 10,
                position: 'insideBottomRight',
              }}
            />

            {/* Confidence Area Upper Bound */}
            <Area
              type="monotone"
              dataKey="confidenceUpper"
              stroke="transparent"
              fill="url(#predictionAreaGrad)"
              name="Confidence Interval"
            />

            {/* Lower Bound Mask */}
            <Area
              type="monotone"
              dataKey="confidenceLower"
              stroke="transparent"
              fill="#111827"
              fillOpacity={0.9}
            />

            {/* Main Predicted Trajectory Line */}
            <Line
              type="monotone"
              dataKey="predictedGlucose"
              name="Projected Glucose"
              stroke="#06b6d4"
              strokeWidth={3}
              dot={(props: any) => {
                const { cx, cy, payload } = props;
                if (payload.isPeak) {
                  return (
                    <g key={`peak-${cx}`}>
                      <circle cx={cx} cy={cy} r={7} fill="#ef4444" stroke="#ffffff" strokeWidth={2} />
                      <circle cx={cx} cy={cy} r={11} fill="transparent" stroke="#ef4444" strokeWidth={1.5} className="animate-ping" />
                    </g>
                  );
                }
                return (
                  <circle
                    key={`dot-${cx}`}
                    cx={cx}
                    cy={cy}
                    r={3.5}
                    fill="#06b6d4"
                    stroke="#0a0e1a"
                    strokeWidth={1.5}
                  />
                );
              }}
              activeDot={{ r: 7, fill: '#06b6d4', stroke: '#ffffff', strokeWidth: 2 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Top 3 Risk Factors Cards */}
      <div className="mt-5 pt-4 border-t border-[#1f293d]">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-cyan-400" />
            Top 3 Risk Drivers (SHAP / Feature Attribution)
          </span>
          <span className="text-[11px] text-slate-400">
            Multimodal Fusion: Static EHR + Wearable CGM + Telemetry
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {prediction.topRiskFactors.map((rf, idx) => (
            <div
              key={rf.id || idx}
              className="bg-[#0a0e1a]/80 border border-[#1f293d] rounded-lg p-3 hover:border-cyan-500/30 transition-all flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold text-slate-200 line-clamp-1">
                    {rf.factor}
                  </span>
                  <span
                    className={`text-[10px] font-bold px-1.5 py-0.5 rounded uppercase ${
                      rf.impact === 'High'
                        ? 'bg-red-950 text-red-300 border border-red-800/60'
                        : rf.impact === 'Medium'
                        ? 'bg-amber-950 text-amber-300 border border-amber-800/60'
                        : 'bg-emerald-950 text-emerald-300 border border-emerald-800/60'
                    }`}
                  >
                    {rf.impact} Impact
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                  {rf.description}
                </p>
              </div>

              <div className="mt-2.5 pt-2 border-t border-[#1f293d]/60 flex items-center justify-between text-xs">
                <span className="text-[11px] text-slate-500">
                  Impact on Peak:
                </span>
                <span
                  className={`font-mono font-bold ${
                    rf.deltaMgDl > 0 ? 'text-red-400' : 'text-emerald-400'
                  }`}
                >
                  {rf.deltaMgDl > 0 ? `+${rf.deltaMgDl}` : rf.deltaMgDl} mg/dL ({rf.contributionPercent}%)
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </motion.section>
  );
}

// Tooltip for 2h Prediction Timeline
function CustomPredictionTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;
  const data = payload[0]?.payload;
  if (!data) return null;

  const glucoseVal = data.predictedGlucose;
  const glucoseColor = getGlucoseColor(glucoseVal);

  return (
    <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-3 shadow-2xl text-xs min-w-[200px]">
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-1 mb-2">
        <span className="font-semibold text-slate-300 flex items-center gap-1">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          {label === 'Now' ? 'Current Baseline' : `Forecast: ${label} (+${data.minutes}m)`}
        </span>
      </div>

      <div className="flex items-baseline justify-between mb-1">
        <span className="text-slate-400">Predicted Glucose:</span>
        <div className="flex items-baseline gap-1">
          <span className="text-lg font-bold font-mono" style={{ color: glucoseColor }}>
            {glucoseVal}
          </span>
          <span className="text-[10px] text-slate-400">mg/dL</span>
        </div>
      </div>

      <div className="flex items-center justify-between text-[11px] text-slate-400 bg-slate-900/60 px-2 py-1 rounded">
        <span>Confidence Range:</span>
        <span className="font-mono text-cyan-300 font-semibold">
          {data.confidenceLower} – {data.confidenceUpper} mg/dL
        </span>
      </div>

      {data.isPeak && (
        <div className="mt-2 text-red-400 text-[11px] font-semibold flex items-center gap-1">
          <Flame className="w-3.5 h-3.5" />
          Predicted Maximum Peak Spike Point
        </div>
      )}
    </div>
  );
}
