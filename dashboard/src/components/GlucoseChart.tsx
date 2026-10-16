'use client';

import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
  CartesianGrid
} from 'recharts';
import {
  Activity,
  Heart,
  Footprints,
  Clock,
  Sparkles,
  TrendingUp,
  Info
} from 'lucide-react';
import { GlucoseReading, SpikePrediction } from '../lib/types';
import { getGlucoseColor } from '../lib/utils';

interface GlucoseChartProps {
  historicalReadings: GlucoseReading[];
  prediction: SpikePrediction;
  patientName: string;
}

interface MergedChartPoint {
  time: string;
  actualGlucose: number | null;
  predictedGlucose: number | null;
  confidenceLower?: number | null;
  confidenceUpper?: number | null;
  heartRate: number | null;
  activity: string | null;
  steps: number | null;
  isPrediction: boolean;
  isNow?: boolean;
}

export default function GlucoseChart({
  historicalReadings,
  prediction,
  patientName
}: GlucoseChartProps) {
  // Time frame selector: 6h, 12h, or 24h
  const [timeRangeHours, setTimeRangeHours] = useState<number>(24);

  // Prepare combined timeline: historical + predicted trajectory
  const chartData = useMemo(() => {
    // 1. Slice historical readings based on selected range (each reading is 30 mins)
    const pointsToKeep = timeRangeHours * 2;
    const slicedHistorical = historicalReadings.slice(-pointsToKeep);

    const merged: MergedChartPoint[] = [];

    // Add historical readings
    slicedHistorical.forEach((reading, idx) => {
      const isLastHistorical = idx === slicedHistorical.length - 1;
      merged.push({
        time: isLastHistorical ? 'Now' : reading.displayTime,
        actualGlucose: reading.glucose,
        predictedGlucose: isLastHistorical ? reading.glucose : null, // Connects line at Now
        heartRate: reading.heartRate,
        activity: reading.activity,
        steps: reading.steps,
        isPrediction: false,
        isNow: isLastHistorical,
      });
    });

    // Add predicted 2-hour points
    prediction.projectedReadings.forEach((predPoint) => {
      merged.push({
        time: predPoint.time,
        actualGlucose: null,
        predictedGlucose: predPoint.predictedGlucose,
        confidenceLower: predPoint.confidenceLower,
        confidenceUpper: predPoint.confidenceUpper,
        heartRate: null,
        activity: 'Projected Rest / Basal',
        steps: null,
        isPrediction: true,
      });
    });

    return merged;
  }, [historicalReadings, prediction, timeRangeHours]);

  return (
    <motion.section
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: 0.05 }}
      className="bg-[#111827] border border-[#1f293d] rounded-xl p-5 shadow-lg flex flex-col justify-between"
      aria-label="Real-time Glucose Telemetry & 2-Hour Forecast"
    >
      {/* Top Header & Range Controls */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-semibold text-white tracking-tight flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-cyan-400" />
              Continuous Glucose Telemetry & 2-Hour Prediction
            </h2>
            <span className="flex items-center gap-1 text-[11px] font-medium px-2 py-0.5 rounded-full bg-cyan-950/60 text-cyan-300 border border-cyan-800/50">
              <Sparkles className="w-3 h-3 text-cyan-400 animate-spin" style={{ animationDuration: '6s' }} />
              Digital Twin Active
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Wearable CGM stream with synchronized cardiac & step telemetry, fused with physiological forecast
          </p>
        </div>

        {/* Range Buttons & Zone Legend */}
        <div className="flex items-center gap-2">
          <div className="bg-[#0a0e1a] p-1 rounded-lg border border-[#1f293d] flex text-xs">
            {[6, 12, 24].map((hours) => (
              <button
                key={hours}
                onClick={() => setTimeRangeHours(hours)}
                className={`px-2.5 py-1 rounded font-medium transition-all ${
                  timeRangeHours === hours
                    ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {hours}h
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Target Zone Badges */}
      <div className="flex flex-wrap items-center gap-4 text-[11px] mb-3 px-1 text-slate-400">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-2 rounded-sm bg-emerald-500/30 border border-emerald-500" />
          <span>Target Range (70–180 mg/dL)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-2 rounded-sm bg-amber-500/30 border border-amber-500" />
          <span>High Range (180–250 mg/dL)</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-2 rounded-sm bg-red-500/30 border border-red-500" />
          <span>Severe Range (&gt;250 / &lt;70 mg/dL)</span>
        </div>
        <div className="flex items-center gap-1.5 ml-auto">
          <span className="w-3 h-0.5 border-b-2 border-dashed border-cyan-400" />
          <span className="text-cyan-300 font-medium">-- 2-Hour Projected Path</span>
        </div>
      </div>

      {/* Recharts Container */}
      <div className="w-full h-[290px] min-h-[290px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartData}
            margin={{ top: 10, right: 15, left: -10, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1f293d" vertical={false} />

            {/* Target & Threshold Reference Zones */}
            <ReferenceArea
              y1={70}
              y2={180}
              fill="#22c55e"
              fillOpacity={0.06}
              stroke="#22c55e"
              strokeOpacity={0.15}
            />
            <ReferenceArea
              y1={180}
              y2={250}
              fill="#f59e0b"
              fillOpacity={0.06}
              stroke="#f59e0b"
              strokeOpacity={0.15}
            />
            <ReferenceArea
              y1={250}
              y2={330}
              fill="#ef4444"
              fillOpacity={0.07}
              stroke="#ef4444"
              strokeOpacity={0.15}
            />
            <ReferenceArea
              y1={40}
              y2={70}
              fill="#ef4444"
              fillOpacity={0.07}
              stroke="#ef4444"
              strokeOpacity={0.15}
            />

            {/* Reference Line at 180 (Spike threshold) */}
            <ReferenceLine
              y={180}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              strokeOpacity={0.6}
              label={{
                value: 'Spike Threshold (180)',
                fill: '#f59e0b',
                fontSize: 10,
                position: 'insideTopRight',
              }}
            />

            {/* Vertical Marker at "Now" */}
            <ReferenceLine
              x="Now"
              stroke="#06b6d4"
              strokeWidth={2}
              label={{
                value: 'NOW (Forecast Zone →)',
                fill: '#06b6d4',
                fontSize: 10,
                position: 'insideTopLeft',
              }}
            />

            <XAxis
              dataKey="time"
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              interval="preserveStartEnd"
              minTickGap={28}
            />

            <YAxis
              domain={[50, 320]}
              stroke="#64748b"
              fontSize={11}
              tickLine={false}
              unit=" mg/dL"
              ticks={[70, 120, 180, 240, 300]}
            />

            <Tooltip content={<CustomGlucoseTooltip />} />

            {/* Solid line for historical CGM data */}
            <Line
              type="monotone"
              dataKey="actualGlucose"
              name="Historical Glucose"
              stroke="#38bdf8"
              strokeWidth={2.5}
              dot={false}
              activeDot={{ r: 5, fill: '#38bdf8', stroke: '#0a0e1a', strokeWidth: 2 }}
              isAnimationActive={true}
              animationDuration={800}
            />

            {/* Dashed line for 2-hour predicted trajectory */}
            <Line
              type="monotone"
              dataKey="predictedGlucose"
              name="2-Hour Projected Glucose"
              stroke="#06b6d4"
              strokeWidth={2.5}
              strokeDasharray="5 5"
              dot={{ r: 3, fill: '#06b6d4', stroke: '#0a0e1a' }}
              activeDot={{ r: 6, fill: '#06b6d4', stroke: '#ffffff', strokeWidth: 2 }}
              isAnimationActive={true}
              animationDuration={800}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Footer Info note */}
      <div className="flex items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-[#1f293d]/80 mt-1">
        <span className="flex items-center gap-1.5">
          <Clock className="w-3.5 h-3.5 text-cyan-400" />
          Last telemetry sync: 15-min CGM epoch
        </span>
        <span className="flex items-center gap-1">
          <Info className="w-3.5 h-3.5 text-slate-500" />
          Bi-LSTM + Gradient Boosted Hybrid Ensemble (94.2% sensitivity)
        </span>
      </div>
    </motion.section>
  );
}

// Custom Tooltip component with complete biometrics
function CustomGlucoseTooltip({ active, payload, label }: any) {
  if (!active || !payload || !payload.length) return null;

  const data: MergedChartPoint = payload[0]?.payload;
  if (!data) return null;

  const glucoseVal = data.isPrediction
    ? data.predictedGlucose
    : data.actualGlucose;

  if (glucoseVal === null || glucoseVal === undefined) return null;

  const glucoseColor = getGlucoseColor(glucoseVal);

  return (
    <div className="bg-[#0f172a] border border-[#334155] rounded-lg p-3 shadow-2xl text-xs min-w-[210px] backdrop-blur-md">
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-1.5 mb-2">
        <span className="font-semibold text-slate-300 flex items-center gap-1">
          <Clock className="w-3 h-3 text-cyan-400" />
          {label === 'Now' ? 'Current Time (Now)' : label}
        </span>
        <span
          className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-medium ${
            data.isPrediction
              ? 'bg-cyan-950 text-cyan-300 border border-cyan-700/50'
              : 'bg-slate-800 text-slate-300'
          }`}
        >
          {data.isPrediction ? '2h AI Forecast' : 'Verified CGM'}
        </span>
      </div>

      {/* Glucose Reading */}
      <div className="flex items-baseline justify-between mb-2">
        <span className="text-slate-400">Glucose:</span>
        <div className="flex items-baseline gap-1">
          <span className="text-base font-bold font-mono" style={{ color: glucoseColor }}>
            {glucoseVal}
          </span>
          <span className="text-[10px] text-slate-400">mg/dL</span>
        </div>
      </div>

      {/* Confidence Bounds for Predictions */}
      {data.isPrediction && data.confidenceLower && data.confidenceUpper && (
        <div className="flex items-center justify-between text-[10px] text-slate-400 mb-2 bg-slate-900/60 px-2 py-1 rounded">
          <span>95% Conf. Interval:</span>
          <span className="font-mono text-cyan-300">
            [{data.confidenceLower} – {data.confidenceUpper}] mg/dL
          </span>
        </div>
      )}

      {/* Wearable Biometrics if present */}
      {!data.isPrediction && (
        <div className="space-y-1.5 pt-1.5 border-t border-slate-800">
          <div className="flex items-center justify-between text-slate-300">
            <span className="flex items-center gap-1.5 text-slate-400">
              <Heart className="w-3 h-3 text-red-400" />
              Heart Rate:
            </span>
            <span className="font-mono font-semibold">{data.heartRate || '--'} bpm</span>
          </div>

          <div className="flex items-center justify-between text-slate-300">
            <span className="flex items-center gap-1.5 text-slate-400">
              <Footprints className="w-3 h-3 text-emerald-400" />
              Activity / Steps:
            </span>
            <span className="font-mono font-semibold">
              {data.steps !== null ? `${data.steps} steps` : '--'}
            </span>
          </div>

          <div className="flex items-center justify-between text-slate-300">
            <span className="flex items-center gap-1.5 text-slate-400">
              <Activity className="w-3 h-3 text-cyan-400" />
              State:
            </span>
            <span className="font-medium text-slate-200">{data.activity || 'Resting'}</span>
          </div>
        </div>
      )}
    </div>
  );
}
