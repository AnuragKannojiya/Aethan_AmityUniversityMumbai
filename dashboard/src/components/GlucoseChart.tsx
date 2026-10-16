'use client';

import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceArea,
  ReferenceLine,
  CartesianGrid,
} from 'recharts';
import { GlucoseReading, TrajectoryPoint } from '../lib/api';

interface Props {
  readings: GlucoseReading[];
  forecast: TrajectoryPoint[];
  simulatedForecast?: TrajectoryPoint[] | null;
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso.slice(11, 16) || iso;
  return d.toTimeString().slice(0, 5);
}

export default function GlucoseChart({ readings, forecast, simulatedForecast }: Props) {
  const { glucoseData, wearableData, nowTime } = useMemo(() => {
    const recent = readings.slice(-72).filter((_, idx, arr) => idx % 2 === 0 || idx === arr.length - 1);

    const gPoints: Array<{
      time: string;
      actual?: number;
      predicted?: number;
      simulated?: number;
    }> = recent.map((r) => ({
      time: formatTime(r.timestamp),
      actual: Math.round(r.glucose_mg_dl),
    }));

    const wPoints: Array<{
      time: string;
      heartRate: number;
      hrv: number;
      steps: number;
      sleepStage: string;
      activity: string;
    }> = recent.map((r) => ({
      time: formatTime(r.timestamp),
      heartRate: Math.round(r.heart_rate_bpm),
      hrv: Math.round(r.hrv_rmssd_ms ?? 38),
      steps: r.steps,
      sleepStage: r.sleep_stage || 'awake',
      activity: r.activity_type,
    }));

    const nowLabel = gPoints.length > 0 ? gPoints[gPoints.length - 1].time : undefined;

    if (gPoints.length > 0 && forecast.length > 0) {
      const lastActual = gPoints[gPoints.length - 1].actual;
      gPoints[gPoints.length - 1].predicted = lastActual;
      if (simulatedForecast && simulatedForecast.length > 0) {
        gPoints[gPoints.length - 1].simulated = lastActual;
      }
    }

    forecast.forEach((f, idx) => {
      gPoints.push({
        time: formatTime(f.time),
        predicted: Math.round(f.glucose),
        simulated:
          simulatedForecast && simulatedForecast[idx]
            ? Math.round(simulatedForecast[idx].glucose)
            : undefined,
      });
    });

    return { glucoseData: gPoints, wearableData: wPoints, nowTime: nowLabel };
  }, [readings, forecast, simulatedForecast]);

  return (
    <div className="space-y-3">
      {/* Primary CGM & 2-Hour Forecast Chart */}
      <div className="w-full h-[265px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={glucoseData} margin={{ top: 8, right: 14, left: -12, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
            <ReferenceArea y1={70} y2={180} fill="#10b981" fillOpacity={0.07} />
            <ReferenceLine
              y={180}
              stroke="#d97706"
              strokeDasharray="4 4"
              label={{
                value: '180 mg/dL Hyperglycemia Threshold',
                position: 'insideTopLeft',
                fill: '#b45309',
                fontSize: 11,
              }}
            />
            <ReferenceLine y={70} stroke="#94a3b8" strokeDasharray="3 3" />
            {nowTime && (
              <ReferenceLine
                x={nowTime}
                stroke="#475569"
                strokeDasharray="2 2"
                label={{ value: 'Now (Forecast →)', position: 'top', fill: '#334155', fontSize: 11 }}
              />
            )}
            <XAxis
              dataKey="time"
              tick={{ fontSize: 11, fill: '#64748b' }}
              tickLine={false}
              axisLine={{ stroke: '#cbd5e1' }}
              minTickGap={26}
            />
            <YAxis
              domain={[55, 340]}
              tick={{ fontSize: 11, fill: '#64748b' }}
              tickLine={false}
              axisLine={false}
              unit=" mg/dL"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#ffffff',
                borderColor: '#cbd5e1',
                borderRadius: '6px',
                fontSize: '12px',
              }}
            />
            <Line
              type="monotone"
              dataKey="actual"
              name="Observed CGM (mg/dL)"
              stroke="#0284c7"
              strokeWidth={2.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              name="2h Digital Twin Forecast"
              stroke="#dc2626"
              strokeWidth={2.5}
              strokeDasharray="5 5"
              dot={{ r: 2.5, fill: '#dc2626' }}
              isAnimationActive={false}
            />
            {simulatedForecast && (
              <Line
                type="monotone"
                dataKey="simulated"
                name="2h Counterfactual (With Intervention)"
                stroke="#059669"
                strokeWidth={2.5}
                strokeDasharray="3 3"
                dot={{ r: 2.5, fill: '#059669' }}
                isAnimationActive={false}
              />
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Synchronized Wearable Stream: HRV (ms) + Heart Rate (bpm) + Steps / 5m */}
      <div className="pt-2 border-t border-slate-100">
        <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-500 mb-1 px-1 gap-2">
          <span className="font-medium text-slate-600">
            Synchronized Wearable IoT Stream (HRV RMSSD, Heart Rate, Steps & Sleep Stage)
          </span>
          <div className="flex items-center gap-3.5">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-violet-600 inline-block" /> HRV RMSSD (ms)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-rose-500 inline-block" /> Heart Rate (bpm)
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-sm bg-sky-400 inline-block" /> Steps / 5m
            </span>
          </div>
        </div>
        <div className="w-full h-[100px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={wearableData} margin={{ top: 4, right: 14, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
              <XAxis
                dataKey="time"
                tick={{ fontSize: 10, fill: '#94a3b8' }}
                tickLine={false}
                axisLine={{ stroke: '#e2e8f0' }}
                minTickGap={26}
              />
              <YAxis
                yAxisId="hr"
                domain={[15, 145]}
                tick={{ fontSize: 10, fill: '#64748b' }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis yAxisId="steps" orientation="right" domain={[0, 400]} hide />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#ffffff',
                  borderColor: '#cbd5e1',
                  borderRadius: '6px',
                  fontSize: '11px',
                }}
              />
              <Bar
                yAxisId="steps"
                dataKey="steps"
                name="Steps (5m)"
                fill="#38bdf8"
                opacity={0.45}
                barSize={6}
                isAnimationActive={false}
              />
              <Line
                yAxisId="hr"
                type="monotone"
                dataKey="heartRate"
                name="Heart Rate (bpm)"
                stroke="#f43f5e"
                strokeWidth={1.6}
                dot={false}
                isAnimationActive={false}
              />
              <Line
                yAxisId="hr"
                type="monotone"
                dataKey="hrv"
                name="HRV RMSSD (ms)"
                stroke="#7c3aed"
                strokeWidth={1.6}
                strokeDasharray="3 2"
                dot={false}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
