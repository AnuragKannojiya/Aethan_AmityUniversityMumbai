/**
 * GlucoTwin - Type Definitions
 * Clinical Digital Twin Platform for Type 2 Diabetes Spike Prediction
 * Amity University Mumbai - Team Aethan
 */

export type RiskLevel = 'Low' | 'Moderate' | 'High' | 'Critical';

export type GlucoseTrend = 
  | 'rising_fast' 
  | 'rising' 
  | 'stable' 
  | 'falling' 
  | 'falling_fast';

export type ActivityType = 
  | 'Resting' 
  | 'Sedentary' 
  | 'Walking' 
  | 'Post-Meal Walk' 
  | 'Exercise' 
  | 'Sleeping'
  | 'Light Activity';

export type ComorbiditySeverity = 'mild' | 'moderate' | 'severe';

export interface Comorbidity {
  name: string;
  severity: ComorbiditySeverity;
  icd10?: string;
}

export interface Patient {
  id: string;
  name: string;
  age: number;
  gender: 'Male' | 'Female' | 'Other';
  city: string;
  avatarUrl?: string;
  bmi: number;
  hba1c: number;
  diabetesDurationYears: number;
  medications: string[];
  comorbidities: Comorbidity[];
  currentRisk: RiskLevel;
  currentRiskScore: number; // 0 - 100
  timeToPotentialSpikeMin: number | null; // minutes or null if no spike
  lastGlucose: number; // mg/dL
  glucoseTrend: GlucoseTrend;
  targetRange: {
    min: number; // standard 70 mg/dL
    max: number; // standard 180 mg/dL
  };
  lastUpdated: string;
}

export interface GlucoseReading {
  timestamp: string; // ISO string
  displayTime: string; // e.g. "14:00"
  glucose: number; // mg/dL
  heartRate: number; // bpm
  steps: number; // step count in 15min window
  activity: ActivityType;
  mealCarbsGrams?: number;
  insulinUnits?: number;
  isHistorical: boolean;
  predictedGlucose?: number;
  confidenceLower?: number;
  confidenceUpper?: number;
}

export interface TrajectoryPoint {
  time: string; // e.g. "16:15", "16:30"
  timestamp: string;
  predictedGlucose: number;
  confidenceLower: number;
  confidenceUpper: number;
  isSpike: boolean;
  spikeThreshold: number; // 180 or 250
}

export interface RiskFactor {
  id: string;
  factor: string;
  impact: 'High' | 'Medium' | 'Low';
  description: string;
  contributionPercent: number; // e.g. 45%
  deltaMgDl: number; // e.g. +42 mg/dL
}

export interface SpikePrediction {
  patientId: string;
  generatedAt: string;
  predictionHorizonMinutes: number; // 120 minutes (2 hours)
  spikeProbability: number; // 0 - 100%
  predictedPeakGlucose: number; // mg/dL
  estimatedTimeToPeakMinutes: number; // e.g. 45
  riskLevel: RiskLevel;
  projectedReadings: TrajectoryPoint[];
  topRiskFactors: RiskFactor[];
}

export type AlertSeverity = 'critical' | 'high' | 'moderate' | 'low';

export type AlertType = 
  | 'impending_spike' 
  | 'sustained_hyperglycemia' 
  | 'hypoglycemia_risk' 
  | 'missed_medication' 
  | 'high_glycemic_variability';

export interface PatientAlert {
  id: string;
  patientId: string;
  severity: AlertSeverity;
  type: AlertType;
  title: string;
  message: string;
  timestamp: string;
  displayTime: string;
  recommendedAction: string;
  isAcknowledged: boolean;
}

export interface DigitalTwinState {
  patientId: string;
  currentGlucose: number;
  glucoseTrendRate: number; // mg/dL per min (e.g. +1.4 mg/dL/min)
  insulinSensitivityFactor: number; // mg/dL per unit
  carbsOnBoard: number; // grams active in GI tract
  insulinOnBoard: number; // units active in circulation
  heartRate: number; // bpm
  hrv: number; // ms (RMSSD)
  activityStatus: ActivityType;
  sleepQuality: {
    isSleeping: boolean;
    durationHours?: number;
    efficiencyPercent?: number;
    stage?: 'Deep' | 'REM' | 'Light' | 'Awake';
    lastNightHours?: number;
  };
  metabolicState: 'Fasting' | 'Postprandial' | 'Basal' | 'Active Burning';
  organStress: {
    pancreasBetaCellLoad: number; // 0 - 100%
    cardiovascularStrain: number; // 0 - 100%
    hepaticGlucoseProduction: number; // 0 - 100%
  };
}

export type InterventionType = 
  | 'take_medication' 
  | 'go_for_walk' 
  | 'eat_balanced_snack';

export interface InterventionOption {
  id: InterventionType;
  title: string;
  description: string;
  icon: string;
  typicalEffect: string;
}

export interface InterventionSimulation {
  type: InterventionType;
  title: string;
  baselineTrajectory: Array<{ time: string; glucose: number }>;
  simulatedTrajectory: Array<{ time: string; glucose: number }>;
  predictedDeltaMgDl: number; // e.g. -46 mg/dL
  clinicalRationale: string;
  estimatedTimeToStabilizeMin: number;
}

export interface DashboardData {
  patient: Patient;
  historicalReadings: GlucoseReading[];
  prediction: SpikePrediction;
  alerts: PatientAlert[];
  twinState: DigitalTwinState;
}
