const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

export interface PatientSummary {
  patient_id: string;
  name: string;
  age: number;
  gender: string;
  city: string;
  bmi: number;
  hba1c: number;
  diabetes_duration_years: number;
  medication: string;
  comorbidities: string[];
  risk_level: 'Low' | 'Moderate' | 'High' | 'Critical';
  last_glucose: number;
  last_updated: string;
}

export interface PatientDetail extends PatientSummary {
  blood_pressure_systolic?: number;
  blood_pressure_diastolic?: number;
  fasting_glucose_baseline?: number;
  egfr?: number;
  creatinine?: number;
  total_cholesterol?: number;
  ldl_cholesterol?: number;
  hdl_cholesterol?: number;
  triglycerides?: number;
  dietary_pattern?: string;
  physical_activity_level?: string;
  sleep_hours_avg?: number;
  stress_level?: string;
  smoking_status?: string;
  family_history_diabetes?: boolean;
  tcf7l2_rs7903146?: string;
  pparg_rs1801282?: string;
  genetic_risk_score?: number;
}

export interface GlucoseReading {
  timestamp: string;
  glucose_mg_dl: number;
  heart_rate_bpm: number;
  hrv_rmssd_ms?: number;
  steps: number;
  sleep_stage: string;
  activity_type: string;
}

export interface TrajectoryPoint {
  time: string;
  glucose: number;
}

export interface FeatureAttribution {
  feature: string;
  stream: string;
  value: string;
  impact_mg_dl: number;
  weight_pct: number;
}

export interface PredictionResponse {
  patient_id: string;
  current_glucose: number;
  predicted_trajectory: TrajectoryPoint[];
  spike_probability: number;
  lstm_probability?: number;
  xgboost_probability?: number;
  risk_level: 'Low' | 'Moderate' | 'High' | 'Critical';
  risk_factors: string[];
  feature_attributions?: FeatureAttribution[];
  recommended_actions: string[];
  time_to_potential_spike: number | null;
  confidence: number;
  model_attention_weights?: number[];
}

export interface AlertItem {
  alert_id: string;
  patient_id: string;
  alert_type: string;
  severity: string;
  message: string;
  timestamp: string;
  recommended_action: string;
}

export interface GlucoseStats {
  time_in_range_pct: number;
  time_above_range_pct: number;
  time_below_range_pct: number;
  mean_glucose: number;
  std_glucose: number;
  coefficient_of_variation: number;
  gmi: number;
}

export interface DashboardPayload {
  patient: PatientDetail;
  recent_readings: GlucoseReading[];
  prediction: PredictionResponse;
  alerts: AlertItem[];
  glucose_stats: GlucoseStats;
}

export interface SimulationParams {
  intervention?: string;
  walk_minutes?: number;
  meal_carbs_g?: number;
  medication_taken?: boolean;
}

export interface SimulationPayload {
  baseline_trajectory: TrajectoryPoint[];
  intervention_trajectory: TrajectoryPoint[];
  expected_improvement: string;
  baseline_peak_mg_dl?: number;
  simulated_peak_mg_dl?: number;
  peak_reduction_mg_dl?: number;
  simulated_tir_pct?: number;
}

export interface SystemOverviewStats {
  total_patients: number;
  active_cgm_streams: number;
  low_risk_count: number;
  moderate_risk_count: number;
  high_risk_count: number;
  critical_risk_count: number;
  mean_population_glucose: number;
  mean_population_hba1c: number;
  avg_time_in_range_pct: number;
  active_alerts_total: number;
  engine_mode: string;
}

export interface ModelMetricsBlock {
  accuracy: number;
  balanced_accuracy: number;
  roc_auc: number;
  pr_auc: number;
  precision: number;
  recall_sensitivity: number;
  specificity: number;
  f1_score: number;
  brier_score: number;
  confusion_matrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

export interface EvaluationMetricsPayload {
  evaluation_cohort_size: number;
  test_spike_ratio_pct: number;
  lstm_model: ModelMetricsBlock;
  xgboost_model: ModelMetricsBlock;
  glucotwin_ensemble: ModelMetricsBlock;
  split_summary?: {
    train_patients: number;
    train_windows: number;
    val_patients: number;
    val_windows: number;
    test_patients: number;
    test_windows: number;
    static_features_count: number;
    temporal_features_count: number;
    xgboost_features_count: number;
    lookback_steps: number;
    horizon_steps: number;
  };
}

export async function fetchPatients(): Promise<PatientSummary[]> {
  const res = await fetch(`${API_BASE}/patients?limit=200`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

export async function fetchDashboard(patientId: string): Promise<DashboardPayload> {
  const res = await fetch(`${API_BASE}/patients/${patientId}/dashboard`);
  if (!res.ok) throw new Error(`Failed to load dashboard (${res.status})`);
  return await res.json();
}

export async function fetchSimulation(
  patientId: string,
  params: SimulationParams
): Promise<SimulationPayload> {
  const res = await fetch(`${API_BASE}/patients/${patientId}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error(`Simulation failed (${res.status})`);
  return await res.json();
}

export async function fetchSystemOverview(): Promise<SystemOverviewStats> {
  const res = await fetch(`${API_BASE}/stats/overview`);
  if (!res.ok) throw new Error(`Overview failed (${res.status})`);
  return await res.json();
}

export async function fetchEvaluationMetrics(): Promise<EvaluationMetricsPayload> {
  const res = await fetch(`${API_BASE}/stats/evaluation`);
  if (!res.ok) throw new Error(`Evaluation metrics failed (${res.status})`);
  return await res.json();
}

