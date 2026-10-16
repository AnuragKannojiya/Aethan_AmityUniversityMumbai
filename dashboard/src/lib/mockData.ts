/**
 * GlucoTwin - Clinical Mock Data
 * Realistic synthetic cohorts modeled after Indian Type 2 Diabetes phenotypes
 * Amity University Mumbai - Team Aethan
 */

import {
  Patient,
  GlucoseReading,
  SpikePrediction,
  PatientAlert,
  DigitalTwinState,
  InterventionSimulation,
  DashboardData,
  InterventionType
} from './types';

// Helper to format ISO and time strings
function getRecentTime(offsetMinutes: number): { iso: string; display: string } {
  const d = new Date();
  d.setMinutes(d.getMinutes() + offsetMinutes);
  const hours = String(d.getHours()).padStart(2, '0');
  const minutes = String(d.getMinutes()).padStart(2, '0');
  return {
    iso: d.toISOString(),
    display: `${hours}:${minutes}`,
  };
}

// 5 Representative Indian T2D Patients
export const MOCK_PATIENTS: Patient[] = [
  {
    id: 'IND-T2D-001',
    name: 'Rajesh Sharma',
    age: 54,
    gender: 'Male',
    city: 'Mumbai, Maharashtra',
    bmi: 28.6,
    hba1c: 8.4,
    diabetesDurationYears: 7,
    medications: [
      'Metformin 1000mg BID (Post-meal)',
      'Glimepiride 2mg OD (Morning)',
      'Teneligliptin 20mg OD (Post-lunch)',
      'Atorvastatin 10mg HS'
    ],
    comorbidities: [
      { name: 'Stage 1 Hypertension', severity: 'moderate', icd10: 'I10' },
      { name: 'Dyslipidemia', severity: 'moderate', icd10: 'E78.5' },
      { name: 'Mild Peripheral Neuropathy', severity: 'mild', icd10: 'E11.40' },
    ],
    currentRisk: 'Critical',
    currentRiskScore: 86,
    timeToPotentialSpikeMin: 35,
    lastGlucose: 218,
    glucoseTrend: 'rising_fast',
    targetRange: { min: 70, max: 180 },
    lastUpdated: 'Just now',
  },
  {
    id: 'IND-T2D-002',
    name: 'Priya Patel',
    age: 48,
    gender: 'Female',
    city: 'Ahmedabad, Gujarat',
    bmi: 25.1,
    hba1c: 7.0,
    diabetesDurationYears: 4,
    medications: [
      'Metformin 500mg BID',
      'Vildagliptin 50mg OD'
    ],
    comorbidities: [
      { name: 'Hypothyroidism (Controlled)', severity: 'mild', icd10: 'E03.9' },
    ],
    currentRisk: 'Low',
    currentRiskScore: 24,
    timeToPotentialSpikeMin: null,
    lastGlucose: 124,
    glucoseTrend: 'stable',
    targetRange: { min: 70, max: 180 },
    lastUpdated: '1 min ago',
  },
  {
    id: 'IND-T2D-003',
    name: 'Sunita Verma',
    age: 62,
    gender: 'Female',
    city: 'New Delhi, Delhi NCR',
    bmi: 31.4,
    hba1c: 9.2,
    diabetesDurationYears: 12,
    medications: [
      'Metformin 1000mg SR BID',
      'Dapagliflozin 10mg OD',
      'Insulin Glargine 18 units HS',
      'Telmisartan 40mg OD'
    ],
    comorbidities: [
      { name: 'Diabetic Nephropathy (Microalbuminuria)', severity: 'moderate', icd10: 'E11.21' },
      { name: 'Hypertension', severity: 'severe', icd10: 'I10' },
      { name: 'Non-Alcoholic Fatty Liver (NAFLD)', severity: 'moderate', icd10: 'K76.0' }
    ],
    currentRisk: 'Critical',
    currentRiskScore: 92,
    timeToPotentialSpikeMin: 20,
    lastGlucose: 242,
    glucoseTrend: 'rising_fast',
    targetRange: { min: 70, max: 180 },
    lastUpdated: 'Just now',
  },
  {
    id: 'IND-T2D-004',
    name: 'Amit Sengupta',
    age: 51,
    gender: 'Male',
    city: 'Kolkata, West Bengal',
    bmi: 27.2,
    hba1c: 7.9,
    diabetesDurationYears: 6,
    medications: [
      'Metformin 850mg BID',
      'Gliclazide 60mg MR OD',
      'Rosuvastatin 10mg OD'
    ],
    comorbidities: [
      { name: 'Essential Hypertension', severity: 'mild', icd10: 'I10' },
      { name: 'Hyperuricemia', severity: 'mild', icd10: 'E79.0' }
    ],
    currentRisk: 'High',
    currentRiskScore: 68,
    timeToPotentialSpikeMin: 55,
    lastGlucose: 178,
    glucoseTrend: 'rising',
    targetRange: { min: 70, max: 180 },
    lastUpdated: '2 mins ago',
  },
  {
    id: 'IND-T2D-005',
    name: 'Dr. Vikram Rao',
    age: 59,
    gender: 'Male',
    city: 'Bengaluru, Karnataka',
    bmi: 23.8,
    hba1c: 6.7,
    diabetesDurationYears: 3,
    medications: [
      'Metformin 500mg OD (Post-dinner)'
    ],
    comorbidities: [
      { name: 'Early Osteopenia', severity: 'mild', icd10: 'M85.8' }
    ],
    currentRisk: 'Low',
    currentRiskScore: 16,
    timeToPotentialSpikeMin: null,
    lastGlucose: 108,
    glucoseTrend: 'stable',
    targetRange: { min: 70, max: 180 },
    lastUpdated: 'Just now',
  }
];

// Generate 24 hours of realistic readings leading up to 'now'
export function generate24HourReadings(patientId: string): GlucoseReading[] {
  const readings: GlucoseReading[] = [];
  const basePatient = MOCK_PATIENTS.find(p => p.id === patientId) || MOCK_PATIENTS[0];
  const now = new Date();

  // Baseline calibration per patient
  const baselineGlucose = 
    patientId === 'IND-T2D-001' ? 145 :
    patientId === 'IND-T2D-002' ? 115 :
    patientId === 'IND-T2D-003' ? 175 :
    patientId === 'IND-T2D-004' ? 135 : 105;

  // Generate 48 points (every 30 mins for 24h)
  for (let i = 47; i >= 0; i--) {
    const timestamp = new Date(now.getTime() - i * 30 * 60 * 1000);
    const hours = timestamp.getHours();
    const minutes = timestamp.getMinutes();
    const displayTime = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;

    // Circadian & Meal profile simulation
    let mealComponent = 0;
    let activity: GlucoseReading['activity'] = 'Resting';
    let steps = 40;
    let heartRate = 72;

    // Breakfast: 8:00 - 9:30 AM
    if (hours === 8 || (hours === 9 && minutes <= 30)) {
      mealComponent += 45;
      activity = 'Sedentary';
      steps = 120;
      heartRate = 78;
    }
    // Lunch: 13:00 - 14:30 PM (Typical high carb Indian lunch: dal, roti, rice, sabzi)
    else if (hours === 13 || (hours === 14 && minutes <= 30)) {
      mealComponent += 65;
      activity = 'Sedentary';
      steps = 90;
      heartRate = 82;
    }
    // Evening snack / chai: 17:00 - 17:30
    else if (hours === 17 && minutes <= 30) {
      mealComponent += 28;
      activity = 'Resting';
      steps = 60;
      heartRate = 74;
    }
    // Dinner: 20:30 - 22:00 PM
    else if (hours === 20 || hours === 21) {
      mealComponent += 55;
      activity = 'Sedentary';
      steps = 150;
      heartRate = 80;
    }
    // Sleep: 23:00 - 06:00
    else if (hours >= 23 || hours < 6) {
      mealComponent -= 15;
      activity = 'Sleeping';
      steps = 0;
      heartRate = 62;
    }
    // Morning walk: 06:30 - 07:15
    else if (hours === 6 && minutes >= 30) {
      mealComponent -= 10;
      activity = 'Walking';
      steps = 1450;
      heartRate = 108;
    }

    // Recent meal bump near the end of the 24-hr window for high-risk patients
    if (i < 4 && (patientId === 'IND-T2D-001' || patientId === 'IND-T2D-003')) {
      mealComponent += (4 - i) * 16; // steep upward trajectory
      activity = 'Sedentary';
      heartRate = 88;
    }

    // Natural variability
    const noise = Math.sin(i * 0.7) * 8;
    let calcGlucose = Math.round(baselineGlucose + mealComponent + noise);

    // If it's the very last reading, match the patient's current lastGlucose
    if (i === 0) {
      calcGlucose = basePatient.lastGlucose;
    }

    // Bounds check
    calcGlucose = Math.max(55, Math.min(330, calcGlucose));

    readings.push({
      timestamp: timestamp.toISOString(),
      displayTime,
      glucose: calcGlucose,
      heartRate: Math.round(heartRate + (Math.random() * 4 - 2)),
      steps: Math.max(0, steps),
      activity,
      isHistorical: true,
    });
  }

  return readings;
}

// Generate Spike Prediction for the next 2 hours (120 minutes)
export function generatePrediction(patientId: string): SpikePrediction {
  const patient = MOCK_PATIENTS.find(p => p.id === patientId) || MOCK_PATIENTS[0];
  const currentG = patient.lastGlucose;
  const horizon = 120; // minutes
  const projectedReadings: SpikePrediction['projectedReadings'] = [];
  const now = new Date();

  // Trajectory dynamics based on patient phenotype
  const isHighRisk = patient.currentRisk === 'Critical' || patient.currentRisk === 'High';
  const peakDelta = isHighRisk 
    ? (patient.id === 'IND-T2D-003' ? 52 : 38) 
    : (patient.currentRisk === 'Moderate' ? 14 : -8);

  const peakMinutes = patient.timeToPotentialSpikeMin || 45;
  const peakGlucose = Math.round(currentG + peakDelta);

  // Generate 8 intervals (+15m, +30m, +45m, +60m, +75m, +90m, +105m, +120m)
  for (let step = 1; step <= 8; step++) {
    const minuteOffset = step * 15;
    const futureDate = new Date(now.getTime() + minuteOffset * 60 * 1000);
    const displayTime = `${String(futureDate.getHours()).padStart(2, '0')}:${String(futureDate.getMinutes()).padStart(2, '0')}`;

    // Curve formula: rise to peak then gradual decay or stabilization
    let predictedVal: number;
    if (minuteOffset <= peakMinutes) {
      const progress = minuteOffset / peakMinutes;
      predictedVal = currentG + (peakDelta * Math.sin((progress * Math.PI) / 2));
    } else {
      const decayProgress = (minuteOffset - peakMinutes) / (horizon - peakMinutes);
      // Returns partially towards baseline
      predictedVal = peakGlucose - (peakDelta * 0.45 * decayProgress);
    }

    const roundedVal = Math.round(predictedVal);
    const spread = Math.round(8 + (minuteOffset / 120) * 16); // uncertainty widens over time

    projectedReadings.push({
      time: displayTime,
      timestamp: futureDate.toISOString(),
      predictedGlucose: roundedVal,
      confidenceLower: Math.max(60, roundedVal - spread),
      confidenceUpper: Math.min(360, roundedVal + spread),
      isSpike: roundedVal >= 180,
      spikeThreshold: 180,
    });
  }

  // Clinical risk factors derived from feature engineering & multimodal fusion
  const topRiskFactors: SpikePrediction['topRiskFactors'] = 
    isHighRisk ? [
      {
        id: 'rf-1',
        factor: 'High-Carbohydrate Postprandial Load',
        impact: 'High',
        description: 'Estimated 72g glycemic load (Indian meal: Chapati, Basmati rice, Chana dal) within past 45 mins.',
        contributionPercent: 48,
        deltaMgDl: 42,
      },
      {
        id: 'rf-2',
        factor: 'Sedentary Post-Meal Period',
        impact: 'High',
        description: 'Under 180 steps recorded in past 60 minutes. Blunted non-insulin dependent GLUT4 translocation.',
        contributionPercent: 32,
        deltaMgDl: 26,
      },
      {
        id: 'rf-3',
        factor: 'Elevated Sympathetic Tone / Stress',
        impact: 'Medium',
        description: 'Elevated resting HR (88 bpm) and reduced HRV (21ms RMSSD) signaling cortisol/adrenergic counter-regulation.',
        contributionPercent: 20,
        deltaMgDl: 16,
      }
    ] : [
      {
        id: 'rf-1',
        factor: 'Balanced Glycemic Index Meal',
        impact: 'Low',
        description: 'High-fiber diet (Moong sprout salad, curd, multigrain) moderating glucose absorption kinetics.',
        contributionPercent: 40,
        deltaMgDl: 8,
      },
      {
        id: 'rf-2',
        factor: 'Active Post-Meal Ambulation',
        impact: 'Low',
        description: '1,420 steps completed post-lunch promoting peripheral muscle glucose uptake.',
        contributionPercent: 35,
        deltaMgDl: -12,
      },
      {
        id: 'rf-3',
        factor: 'Optimal Medication Adherence',
        impact: 'Low',
        description: 'Metformin active therapeutic window maintained; liver gluconeogenesis properly curbed.',
        contributionPercent: 25,
        deltaMgDl: -10,
      }
    ];

  return {
    patientId,
    generatedAt: now.toISOString(),
    predictionHorizonMinutes: horizon,
    spikeProbability: patient.currentRiskScore,
    predictedPeakGlucose: peakGlucose,
    estimatedTimeToPeakMinutes: peakMinutes,
    riskLevel: patient.currentRisk,
    projectedReadings,
    topRiskFactors,
  };
}

// Generate Realistic Active Clinical Alerts
export function generateAlerts(patientId: string): PatientAlert[] {
  const patient = MOCK_PATIENTS.find(p => p.id === patientId) || MOCK_PATIENTS[0];
  const now = new Date();

  if (patient.id === 'IND-T2D-001') {
    return [
      {
        id: 'alt-001',
        patientId,
        severity: 'critical',
        type: 'impending_spike',
        title: 'Severe Glucose Spike Predicted in 35 Mins',
        message: 'Digital Twin models 86% probability of postprandial glucose crossing 245 mg/dL. Current trajectory: +1.6 mg/dL/min.',
        timestamp: new Date(now.getTime() - 4 * 60 * 1000).toISOString(),
        displayTime: '4 mins ago',
        recommendedAction: 'Advise 15-minute moderate brisk walk immediately to stimulate GLUT4 uptake, or administer prescribed rapid-acting correction if ordered.',
        isAcknowledged: false,
      },
      {
        id: 'alt-002',
        patientId,
        severity: 'high',
        type: 'sustained_hyperglycemia',
        title: 'High Glycemic Variability (TIR: 62%)',
        message: 'Time in Range (70-180 mg/dL) dipped below clinical target of 70% over the last 24 hours with 3 distinct excursions.',
        timestamp: new Date(now.getTime() - 42 * 60 * 1000).toISOString(),
        displayTime: '42 mins ago',
        recommendedAction: 'Review carbohydrate distribution at dinner; consider titration of Teneligliptin with endocrinologist.',
        isAcknowledged: false,
      },
      {
        id: 'alt-003',
        patientId,
        severity: 'moderate',
        type: 'missed_medication',
        title: 'Possible Delayed Post-Lunch Teneligliptin',
        message: 'No wearable confirmation or log for scheduled 2:00 PM oral dose. Baseline insulin secretion deficit detected.',
        timestamp: new Date(now.getTime() - 95 * 60 * 1000).toISOString(),
        displayTime: '1 hr ago',
        recommendedAction: 'Trigger patient companion app reminder to confirm ingestion of prescribed DPP-4 inhibitor.',
        isAcknowledged: true,
      }
    ];
  }

  if (patient.id === 'IND-T2D-003') {
    return [
      {
        id: 'alt-101',
        patientId,
        severity: 'critical',
        type: 'impending_spike',
        title: 'Severe Postprandial Excursion (>270 mg/dL Expected)',
        message: 'Patient currently at 242 mg/dL. Renal threshold for glucose spillage (glucosuria) exceeded. Nephropathy strain alert.',
        timestamp: new Date(now.getTime() - 2 * 60 * 1000).toISOString(),
        displayTime: '2 mins ago',
        recommendedAction: 'Immediate clinical review. Hydration with 500ml water advised; check for urine ketones if symptoms present.',
        isAcknowledged: false,
      },
      {
        id: 'alt-102',
        patientId,
        severity: 'high',
        type: 'sustained_hyperglycemia',
        title: 'Nocturnal Basal Insufficiency',
        message: 'Overnight glucose averaged 194 mg/dL. Dawn phenomenon noted with +45 mg/dL spike between 04:30 and 06:30.',
        timestamp: new Date(now.getTime() - 280 * 60 * 1000).toISOString(),
        displayTime: '4 hrs ago',
        recommendedAction: 'Re-evaluate evening Insulin Glargine dosing (currently 18 units). Target fasting glucose < 130 mg/dL.',
        isAcknowledged: false,
      }
    ];
  }

  if (patient.id === 'IND-T2D-004') {
    return [
      {
        id: 'alt-201',
        patientId,
        severity: 'high',
        type: 'impending_spike',
        title: 'Approaching Upper Target Limit (180 mg/dL)',
        message: 'Current glucose 178 mg/dL with steady positive velocity (+0.9 mg/dL/min). Spike above 210 mg/dL anticipated in 55 mins.',
        timestamp: new Date(now.getTime() - 12 * 60 * 1000).toISOString(),
        displayTime: '12 mins ago',
        recommendedAction: 'Encourage 10-15 minute walk. Delay any high-glycemic snack consumption.',
        isAcknowledged: false,
      },
      {
        id: 'alt-202',
        patientId,
        severity: 'moderate',
        type: 'high_glycemic_variability',
        title: 'Moderate Afternoon Excursion',
        message: 'Standard deviation of glucose in last 12 hours is 44 mg/dL (target < 36 mg/dL).',
        timestamp: new Date(now.getTime() - 110 * 60 * 1000).toISOString(),
        displayTime: '1.8 hrs ago',
        recommendedAction: 'Log meal composition in GlucoTwin mobile module.',
        isAcknowledged: false,
      }
    ];
  }

  // Stable / Well-controlled patients (IND-T2D-002, IND-T2D-005)
  return [
    {
      id: 'alt-301',
      patientId,
      severity: 'low',
      type: 'high_glycemic_variability',
      title: 'Stable Euglycemic State Maintained',
      message: 'Time in Range (70-180 mg/dL) is 94% over the past 24 hours. Minimal glycemic variability observed.',
      timestamp: new Date(now.getTime() - 35 * 60 * 1000).toISOString(),
      displayTime: '35 mins ago',
      recommendedAction: 'Maintain current regimen and daily 30-minute aerobic regimen.',
      isAcknowledged: true,
    }
  ];
}

// Digital Twin Organ & Physiological State
export function generateTwinState(patientId: string): DigitalTwinState {
  const patient = MOCK_PATIENTS.find(p => p.id === patientId) || MOCK_PATIENTS[0];
  const isHigh = patient.currentRisk === 'Critical' || patient.currentRisk === 'High';

  return {
    patientId,
    currentGlucose: patient.lastGlucose,
    glucoseTrendRate: isHigh ? +1.6 : (patient.currentRisk === 'Moderate' ? +0.7 : -0.2),
    insulinSensitivityFactor: patient.id === 'IND-T2D-003' ? 18 : 28, // mg/dL per unit
    carbsOnBoard: isHigh ? 64 : 18, // grams in digestive tract
    insulinOnBoard: patient.id === 'IND-T2D-003' ? 3.4 : 1.2, // units active
    heartRate: isHigh ? 86 : 72,
    hrv: isHigh ? 22 : 46, // ms
    activityStatus: isHigh ? 'Sedentary' : 'Light Activity',
    sleepQuality: {
      isSleeping: false,
      lastNightHours: 6.8,
      efficiencyPercent: isHigh ? 74 : 88,
      stage: 'Awake',
    },
    metabolicState: isHigh ? 'Postprandial' : 'Basal',
    organStress: {
      pancreasBetaCellLoad: isHigh ? 84 : 38,
      cardiovascularStrain: isHigh ? 68 : 24,
      hepaticGlucoseProduction: isHigh ? 76 : 30,
    }
  };
}

// What-If Simulation Generator
export function simulateIntervention(
  patientId: string, 
  intervention: InterventionType
): InterventionSimulation {
  const patient = MOCK_PATIENTS.find(p => p.id === patientId) || MOCK_PATIENTS[0];
  const currentG = patient.lastGlucose;
  const now = new Date();

  // Baseline 2-hour trajectory without intervention (spiking or current trend)
  const baselineTrajectory: Array<{ time: string; glucose: number }> = [];
  const simulatedTrajectory: Array<{ time: string; glucose: number }> = [];

  const baselinePeakDelta = patient.currentRisk === 'Critical' ? 44 : 28;

  // Intervention effects
  let deltaEffect = 0;
  let title = '';
  let clinicalRationale = '';
  let timeToStabilize = 45;

  switch (intervention) {
    case 'take_medication':
      title = 'Administer Prescribed Medication (Teneligliptin / Metformin Titration)';
      deltaEffect = -38;
      timeToStabilize = 60;
      clinicalRationale = 
        'DPP-4 inhibition enhances endogenous GLP-1 longevity, stimulating glucose-dependent insulin secretion from pancreatic beta cells while concurrently repressing glucagon output from alpha cells. Predicted reduction of peak amplitude by 38 mg/dL.';
      break;

    case 'go_for_walk':
      title = '20-Minute Moderate Post-Meal Brisk Walk (3.5 - 4.0 km/h)';
      deltaEffect = -48;
      timeToStabilize = 35;
      clinicalRationale = 
        'Skeletal muscle contraction triggers AMPK activation and translocates GLUT4 glucose transporters directly to cell membranes completely independent of insulin. This rapidly drains active circulating glucose, blunting peak postprandial spike by ~48 mg/dL.';
      break;

    case 'eat_balanced_snack':
      title = 'Consume High-Fiber & Protein Snack (Sprouts / Roasted Chana)';
      deltaEffect = -18;
      timeToStabilize = 50;
      clinicalRationale = 
        'Introduction of soluble fiber and proteins slows gastric emptying rate (decreased pyloric flow) and blunts rapid monosaccharide absorption in the duodenum, flattening the postprandial glycemic surge.';
      break;
  }

  // Generate 8 data points (15m increments)
  for (let i = 0; i <= 8; i++) {
    const mins = i * 15;
    const futureDate = new Date(now.getTime() + mins * 60 * 1000);
    const timeLabel = `${String(futureDate.getHours()).padStart(2, '0')}:${String(futureDate.getMinutes()).padStart(2, '0')}`;

    if (i === 0) {
      baselineTrajectory.push({ time: timeLabel, glucose: currentG });
      simulatedTrajectory.push({ time: timeLabel, glucose: currentG });
      continue;
    }

    // Baseline calculation (rises then plateaus)
    const baselineProgress = Math.min(1, mins / 45);
    const baseG = Math.round(currentG + baselinePeakDelta * Math.sin((baselineProgress * Math.PI) / 2));
    baselineTrajectory.push({ time: timeLabel, glucose: baseG });

    // Simulated trajectory with intervention active
    const interventionProgress = Math.min(1, mins / timeToStabilize);
    const reduction = deltaEffect * Math.sin((interventionProgress * Math.PI) / 2);
    const simG = Math.max(75, Math.round(baseG + reduction));
    simulatedTrajectory.push({ time: timeLabel, glucose: simG });
  }

  return {
    type: intervention,
    title,
    baselineTrajectory,
    simulatedTrajectory,
    predictedDeltaMgDl: deltaEffect,
    clinicalRationale,
    estimatedTimeToStabilizeMin: timeToStabilize,
  };
}

// Complete Dashboard Aggregator
export function getMockDashboardData(patientId: string): DashboardData {
  const patient = MOCK_PATIENTS.find(p => p.id === patientId) || MOCK_PATIENTS[0];
  const historicalReadings = generate24HourReadings(patient.id);
  const prediction = generatePrediction(patient.id);
  const alerts = generateAlerts(patient.id);
  const twinState = generateTwinState(patient.id);

  return {
    patient,
    historicalReadings,
    prediction,
    alerts,
    twinState,
  };
}
