# 🏥 GlucoTwin — AI-Powered Digital Twin for Type 2 Diabetes

<div align="center">

**Predicting Glucose Spikes 2 Hours in Advance Using Multi-Modal Deep Learning**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Digital Twin Challenge 2026 — Reimagining & Reforming Healthcare in India Summit*

</div>

---

## 📋 Table of Contents

- [Team Details](#-team-details)
- [Project Title](#-project-title)
- [Problem Statement & Healthcare Use Case](#-problem-statement--healthcare-use-case)
- [Solution Architecture](#-solution-architecture)
- [Technical Stack](#-technical-stack)
- [AI/ML Model Details](#-aiml-model-details)
- [Dataset Strategy](#-dataset-strategy)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the Project](#-running-the-project)
- [Dashboard Preview](#-dashboard-preview)
- [Evaluation Metrics](#-evaluation-metrics)
- [Demo Video](#-demo-video)
- [License](#-license)

---

## 👥 Team Details

| | Details |
|---|---|
| **Team Name** | Aethan |
| **College** | Amity University Mumbai |
| **Challenge** | Digital Twin Challenge 2026 |
| **Summit** | Reimagining & Reforming Healthcare in India Summit 2026, Bengaluru |

---

## 🏷️ Project Title

**GlucoTwin: A Real-Time Digital Twin Platform for Predictive Glucose Management in Type 2 Diabetes**

---

## 🎯 Problem Statement & Healthcare Use Case

### The Problem

India is the **diabetes capital of the world** with over 101 million diagnosed cases (ICMR 2023). Type 2 Diabetes Mellitus (T2DM) accounts for ~95% of all diabetes cases in India. The current reactive model of diabetes management — periodic HbA1c tests and clinic visits — fails to capture the dynamic, real-time glycemic variability that drives acute complications like:

- **Hyperglycemic spikes** (>250 mg/dL) causing diabetic ketoacidosis
- **Hypoglycemic episodes** (<70 mg/dL) leading to seizures, falls, and cardiac events
- **Glycemic variability** accelerating microvascular damage (retinopathy, nephropathy, neuropathy)

### Our Solution: GlucoTwin

GlucoTwin is a **Digital Twin proof-of-concept** that creates a dynamic, virtual replica of a Type 2 Diabetes patient by fusing two data streams:

1. **Static/Historical Data (EHR):** Demographics, past diagnoses, lab results (HbA1c, lipid profile, renal function), medications, and comorbidities
2. **Dynamic/Real-Time Data (Wearables):** Continuous glucose monitoring (CGM), heart rate variability, physical activity, sleep patterns, and stress indices

The Digital Twin processes these streams through a **multi-modal LSTM + XGBoost ensemble** to:
- ✅ **Predict glucose spikes 2 hours in advance** with >85% accuracy
- ✅ **Assess patient-specific risk** using clinical rule integration
- ✅ **Simulate interventions** (medication, exercise, dietary changes) via counterfactual analysis
- ✅ **Alert clinicians** through a real-time doctor-facing dashboard

### Healthcare Impact for India

- **Proactive diabetes management** replacing reactive clinic-based monitoring
- **Personalized treatment protocols** adapted to Indian dietary patterns (high-carb vegetarian diets, meal timing)
- **Reduced emergency hospitalizations** from preventable hyper/hypoglycemic events
- **Scalable to India's 101M+ diabetic population** using consumer wearables + smartphone apps

---

## 📐 Solution Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES (Synthetic)                      │
├──────────────────┬───────────────────────────────────────────────┤
│  Synthea-Style   │    Wearable/IoT Simulation                   │
│  EHR Generator   │    CGM + Heart Rate + Steps + Sleep          │
│  200 Indian T2D  │    5-min intervals, 30 days                  │
│  Patients        │    1,728,000 time-series readings             │
└────────┬─────────┴─────────────────┬─────────────────────────────┘
         │                           │
         ▼                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                                 │
├──────────────────┬─────────────────┬─────────────────────────────┤
│  EHR Processor   │  CGM Processor  │  Data Fusion Engine         │
│  • Clean/Validate│  • 2h Rolling   │  • Broadcast Merge          │
│  • Risk Stratify │    Glucose Stats│  • Sliding Window Sequences │
│  • Encode/Scale  │  • HR Metrics   │  • Patient-Level Splits     │
│  • 46 Features   │  • Activity/    │  • Train/Val/Test (.npz)    │
│                  │    Sleep Metrics│                              │
│                  │  • 54 Features  │                              │
└────────┬─────────┴────────┬────────┴─────────────────────────────┘
         │                  │
         ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    AI/ML ENGINE                                  │
├──────────────────────────┬───────────────────────────────────────┤
│  Temporal Branch (LSTM)  │  Tabular Branch (XGBoost)            │
│  • 2-Layer Bi-LSTM       │  • Gradient Boosted Trees            │
│  • Self-Attention Mech.  │  • 180 Estimators, Depth 6           │
│  • Static Feature MLP    │  • Rolling Summary Statistics        │
│  • Hidden Dim: 128       │  • Auto Class Weighting              │
│  • Weight: 0.6           │  • Weight: 0.4                       │
├──────────────────────────┴───────────────────────────────────────┤
│              GlucoTwin Ensemble (Weighted Average)                │
│              Threshold-Calibrated Spike Prediction                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                    DIGITAL TWIN ENGINE                            │
├──────────────────┬─────────────────┬─────────────────────────────┤
│  Twin State Mgr  │ Risk Calculator │  Patient Simulator          │
│  • Per-Patient   │ • Composite     │  • Insulin Response Models  │
│    Virtual Model │   Risk Scoring  │  • Exercise Glucose Uptake  │
│  • Real-Time     │ • Clinical Rule │  • Meal Excursion Models    │
│    State Updates │   Integration   │  • Counterfactual Analysis  │
│  • Alert Engine  │ • ADA/RSSDI     │  • What-If Scenarios        │
│                  │   Guidelines    │                              │
└────────┬─────────┴────────┬────────┴─────────────────────────────┘
         │                  │
         ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                    API + DASHBOARD                                │
├──────────────────────────┬───────────────────────────────────────┤
│  FastAPI Backend         │  Next.js Doctor Dashboard             │
│  • RESTful Endpoints     │  • Real-Time Glucose Trends           │
│  • Patient CRUD          │  • Risk Assessment Gauge              │
│  • Prediction API        │  • 2-Hour Prediction Timeline         │
│  • Simulation API        │  • Clinical Alert Panel               │
│  • Demo Mode Fallback    │  • Digital Twin Visualization         │
│  • Port 8000             │  • What-If Simulation Modal           │
│                          │  • Dark Medical Theme                 │
│                          │  • Port 3000                          │
└──────────────────────────┴───────────────────────────────────────┘
```

---

## 🔧 Technical Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Data Generation** | NumPy, Pandas, Faker | Synthetic EHR + CGM data (Synthea-style) |
| **Data Processing** | Pandas, Scikit-learn, SciPy | Feature engineering, normalization, sequence creation |
| **Deep Learning** | PyTorch 2.0+ | LSTM with temporal self-attention for glucose prediction |
| **Gradient Boosting** | XGBoost 2.0+ | Tabular risk scoring with rolling summary statistics |
| **API Backend** | FastAPI, Uvicorn, Pydantic | RESTful prediction and simulation API |
| **Frontend** | Next.js 14, React, TypeScript | Doctor-facing clinical dashboard |
| **Visualization** | Recharts, Framer Motion, Tailwind CSS | Interactive charts, animations, medical UI |
| **Testing** | unittest, pytest | 32 unit/integration tests across all modules |

---

## 🧠 AI/ML Model Details

### Multi-Modal Architecture

GlucoTwin employs a **dual-branch ensemble** that fuses temporal and tabular learning:

#### Branch 1: GlucoseLSTM (Weight: 0.6)
- **Architecture:** 2-layer stacked LSTM (hidden_dim=128) with self-attention mechanism
- **Input:** 24-step temporal sequences (2h lookback × 54 dynamic features) + 53 static EHR features via MLP fusion
- **Attention:** Self-attention weights over LSTM outputs to learn which timesteps are most predictive of future spikes
- **Optimizer:** Adam (lr=1e-3) with cosine annealing scheduler
- **Loss:** Weighted binary cross-entropy (class-balanced for spike/no-spike imbalance)
- **Regularization:** Dropout (0.3), gradient clipping (max_norm=1.0), weight decay (1e-5)

#### Branch 2: GlucoseXGBoost (Weight: 0.4)
- **Architecture:** 180 gradient-boosted trees (max_depth=6, histogram-based)
- **Input:** Flattened temporal summary statistics (last timestep, mean, std, min, max, slope) + static features (485 total tabular features)
- **Optimization:** Auto class weighting, early stopping (15 rounds), subsample=0.8

#### Ensemble: GlucoTwinEnsemble
- **Strategy:** Calibrated weighted average (0.6 × LSTM + 0.4 × XGBoost)
- **Output:** Spike probability (0-1), risk level classification, top contributing features

### Feature Engineering (107 Multimodal Features)

**54 Dynamic Features (Real-Time Wearable / IoT Time-Series):**
- **Continuous Glucose Monitors (CGM):** Multi-scale glucose statistics (5m, 15m, 30m, 1h, 2h rolling windows), rate of change, acceleration, momentum, volatility index, Time-in-Range (70-180 mg/dL), coefficient of variation
- **Heart Rate Variability (HRV) & Cardiac:** Beat-to-beat `hrv_rmssd_ms`, 1h SDNN variability (`hr_variability`), resting nocturnal HR, autonomic recovery delta (`hr_recovery_delta`)
- **Sleep Stages:** Ultradian 90-minute sleep stages (`Deep`, `REM`, `Light`, `Awake`), sleep duration, sleep efficiency, deep/REM percentages
- **Step Count & Activity:** 5-min step counts (`steps_last_5min`), hourly step velocity, active/sedentary minutes, 2h step trend
- **Circadian & Meal Kinetics:** Sin/cos time-of-day and day-of-week encodings, dawn phenomenon flags, meal proximity & absorption decay

**53 Static Features (Historical Electronic Health Records):**
- **Demographics:** Age group buckets, gender, metropolitan city, height, weight, BMI category (ICMR Asian Indian cutoffs)
- **Past Diagnoses & Comorbidities:** T2D duration, family history, ICD-10 comorbidities (Hypertension, Dyslipidemia, CKD, Retinopathy, Neuropathy, CAD, NAFLD)
- **Lab Results:** HbA1c tier, fasting blood glucose, eGFR stage, serum creatinine, systolic/diastolic BP, lipid panel (total cholesterol, LDL, HDL, triglycerides, TG/HDL ratio)
- **Genetic Markers (South Asian T2D GWAS Loci):** `TCF7L2 rs7903146` (CC/CT/TT risk alleles), `PPARG rs1801282` (Pro12Ala protective variant), `KCNJ11 rs5219` (E23K beta-cell variant), `SLC30A8 rs13266634` (zinc transporter), `FTO rs9939609` (adiposity locus), `HLA-DQ` risk haplotype, and composite **Polygenic Risk Score (`genetic_risk_score`)**
- **Medications & Indian Phenotype Factors:** Medication complexity score (Metformin, Sulfonylureas, DPP-4i, SGLT2i, Insulin), vegetarian protein adequacy, South Asian thin-fat visceral adiposity score, urban stress index

---

## 📊 Dataset Strategy

### Privacy Compliance

We strictly adhere to **DPDP Act (India)** and **HIPAA** requirements by using **exclusively synthetic and anonymized data**:

| Dataset | Type | Description |
|---------|------|-------------|
| **EHR Records** | Synthetic (Synthea-style) | 200 Indian T2D patients with demographics, past diagnoses, lab panels, genetic markers (`TCF7L2`, `PPARG`, `KCNJ11`, `SLC30A8`, `FTO`, `HLA-DQ`), and medications |
| **CGM & Wearable Readings** | Synthetic Time-Series | 1,728,000 readings (5-min intervals, 30 days, 200 patients) with CGM glucose, HRV (`hrv_rmssd_ms`), heart rate, sleep stages, step counts, skin temp, and stress |
| **Glucose Events** | Derived | 21,900 annotated spike/hypo/rapid-drop events |
| **Medication Schedules** | Synthetic | Per-patient dosing times and adherence rates |

### Physiological Realism

Our synthetic data engine generates clinically plausible patterns:
- **Post-prandial glucose spikes** correlated with Indian meal timings (breakfast 8am, lunch 1pm, chai 5pm, dinner 8:30pm)
- **Dawn phenomenon** (4-8am hepatic glucose output surge)
- **Exercise-induced glucose drops** with GLUT4-mediated muscle uptake
- **Nocturnal patterns** following ultradian 90-minute sleep cycles
- **Patient-specific baselines** correlated with HbA1c (ADA eAG formula: eAG = 28.7 × HbA1c − 46.7)
- **Indian demographic realism** (cities, dietary patterns, medication preferences per RSSDI guidelines)

---

## 📂 Project Structure

```
Aethan_AmityUniversityMumbai/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .gitignore
├── architecture/
│   ├── architecture-diagram.pdf        # System architecture diagram
│   └── presentation.pdf               # Project presentation
├── data/
│   ├── generate_synthetic.py           # Master data generation script
│   ├── raw/                            # Raw synthetic datasets
│   │   ├── ehr_patients.csv            # 200 patient EHR records
│   │   ├── cgm_readings.csv            # 1.7M CGM time-series readings
│   │   ├── glucose_events.csv          # 20,930 annotated events
│   │   └── medications_schedule.csv    # Per-patient medication timing
│   └── processed/                      # ML-ready processed data
│       ├── train_data.npz              # Training split (140 patients)
│       ├── val_data.npz                # Validation split (20 patients)
│       ├── test_data.npz               # Test split (40 patients)
│       ├── feature_metadata.json       # Feature schema & statistics
│       └── models/                     # Trained model checkpoints
├── src/
│   ├── data_pipeline/
│   │   ├── ehr_processor.py            # EHR data cleaning & encoding
│   │   ├── cgm_processor.py            # CGM time-series processing
│   │   └── data_fusion.py             # Multi-modal data fusion
│   ├── feature_engineering/
│   │   ├── static_features.py          # EHR feature extraction (46 features)
│   │   ├── temporal_features.py        # CGM temporal features (54 features)
│   │   └── feature_store.py           # Unified feature management
│   ├── models/
│   │   ├── lstm_model.py               # PyTorch LSTM with self-attention
│   │   ├── xgboost_model.py            # XGBoost risk classifier
│   │   ├── ensemble.py                # GlucoTwin ensemble predictor
│   │   └── train.py                   # Training orchestrator
│   ├── digital_twin/
│   │   ├── twin_engine.py              # Core Digital Twin engine
│   │   ├── risk_calculator.py          # Clinical risk scoring
│   │   └── simulator.py              # Intervention simulator
│   └── api/
│       ├── main.py                     # FastAPI application
│       ├── routes.py                   # API endpoint definitions
│       ├── schemas.py                  # Pydantic data models
│       └── service.py                 # Business logic service layer
├── dashboard/                          # Next.js React dashboard
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx                # Main dashboard page
│   │   │   ├── layout.tsx              # Root layout (dark medical theme)
│   │   │   └── globals.css             # Theme CSS variables
│   │   ├── components/
│   │   │   ├── PatientOverview.tsx      # Patient vitals card
│   │   │   ├── GlucoseChart.tsx        # 24h glucose trend + prediction
│   │   │   ├── RiskGauge.tsx           # Circular risk gauge (0-100%)
│   │   │   ├── PredictionTimeline.tsx  # 2h prediction with confidence
│   │   │   ├── AlertPanel.tsx          # Clinical alert list
│   │   │   └── TwinVisualization.tsx   # Digital Twin state + What-If
│   │   └── lib/
│   │       ├── api.ts                  # API client
│   │       ├── mockData.ts             # Standalone mock data
│   │       ├── types.ts                # TypeScript type definitions
│   │       └── utils.ts               # Utility functions
│   └── package.json
└── tests/
    ├── test_data_pipeline.py           # EHR processor tests (10 tests)
    ├── test_cgm_and_fusion.py          # CGM + fusion tests (10 tests)
    └── test_api.py                    # API endpoint tests (12 tests)
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/Aethan_AmityUniversityMumbai.git
cd Aethan_AmityUniversityMumbai
```

### 2. Python Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Generate Synthetic Data
```bash
python data/generate_synthetic.py
```
This generates ~1.7M CGM readings + 200 EHR records in `data/raw/`.

### 4. Process Data & Create ML-Ready Splits
The data pipeline automatically runs during training, or you can trigger it via the API startup.

### 5. Train ML Models
```bash
python -m src.models.train --epochs 10
```
Trains LSTM + XGBoost ensemble. Models saved to `data/processed/models/`.

### 6. Dashboard Setup
```bash
cd dashboard
npm install
```

---

## ▶️ Running the Project

### Option 1: Full Stack (API + Dashboard)

**Terminal 1 — Start the API server:**
```bash
cd Aethan_AmityUniversityMumbai
source .venv/bin/activate
python -m src.api.main
# API runs on http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

**Terminal 2 — Start the Dashboard:**
```bash
cd Aethan_AmityUniversityMumbai/dashboard
npm run dev
# Dashboard runs on http://localhost:3000
```

### Option 2: Dashboard Only (Mock Data)
The dashboard works standalone with built-in mock data — no API needed:
```bash
cd dashboard
npm run dev
```

### Option 3: API Only
```bash
python -m src.api.main
# Visit http://localhost:8000/docs for interactive API documentation
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/patients` | List all patients with filters |
| `GET` | `/api/patients/{id}` | Patient clinical profile |
| `GET` | `/api/patients/{id}/readings` | CGM telemetry (configurable window) |
| `GET` | `/api/patients/{id}/predict` | 2-hour glucose prediction + risk |
| `POST` | `/api/patients/{id}/simulate` | Intervention simulation |
| `GET` | `/api/patients/{id}/alerts` | Active clinical alerts |
| `GET` | `/api/patients/{id}/dashboard` | Complete dashboard payload |
| `GET` | `/api/patients/{id}/glucose-stats` | Time-in-range statistics |
| `GET` | `/api/stats/overview` | System-wide analytics |

---

## 🖥️ Dashboard Preview

The doctor-facing dashboard provides a comprehensive view of each patient's Digital Twin:

### Dashboard Features
1. **Patient Overview** — Demographics, vitals, medications, comorbidities, and risk level
2. **24h Glucose Trend Chart** — Real-time CGM data with color-coded zones (green: 70-180, yellow: 180-250, red: >250 mg/dL)
3. **Risk Assessment Gauge** — Circular gauge showing spike probability with color-coded severity
4. **2-Hour Prediction Timeline** — Forecasted glucose trajectory with confidence bounds
5. **Clinical Alert Panel** — Severity-ranked alerts with recommended actions
6. **Digital Twin Visualization** — Live twin state with organ-level metrics and **What-If simulation** for testing interventions

---

## 📈 Evaluation Metrics

### Model Performance (Test Set — 40 patients, 57,480 sequences)

| Metric | LSTM (0.6) | XGBoost (0.4) | Ensemble |
|--------|-----------|---------------|----------|
| **ROC-AUC** | 0.9915 | 0.9941 | **0.9933** |
| **PR-AUC** | 0.9960 | 0.9972 | **0.9969** |
| **Accuracy** | 94.84% | 95.55% | **95.42%** |
| **Sensitivity** | 95.16% | 95.32% | **95.56%** |
| **Specificity** | 94.16% | 96.03% | **95.10%** |
| **F1-Score** | 0.9616 | 0.9668 | **0.9659** |
| **Brier Score** | 0.0361 | 0.0311 | **0.0320** |

> **Key Result:** The ensemble achieves **99.33% ROC-AUC** and **95.56% sensitivity** on the held-out test cohort, meaning it correctly identifies 95.56% of glucose spikes 2 hours before they occur with a false positive rate of only 4.90%.

### Test Suite
- **32 tests** across all modules — 100% pass rate
- Coverage: Data pipeline, feature engineering, API endpoints, model inference

---

## 🎥 Demo Video

> **[20-Minute Video Demonstration]** — *(Link to be added)*
>
> The video covers:
> 1. Problem statement and healthcare context
> 2. Architecture walkthrough
> 3. Data pipeline demonstration
> 4. Model training and evaluation
> 5. Live dashboard demo with real-time predictions
> 6. What-If intervention simulation
> 7. Clinical impact and future roadmap

---

## 🔮 Future Roadmap

- **Real wearable integration** via Apple HealthKit / Google Fit APIs
- **Federated learning** for privacy-preserving model training across hospitals
- **Multi-condition Digital Twin** extending to hypertension and cardiovascular risk
- **FHIR/HL7 interoperability** for EHR integration with Indian hospital systems (ABDM)
- **Mobile companion app** for patient self-monitoring
- **Explainable AI (XAI)** dashboard showing SHAP-based feature attribution

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Happiest Health** for organizing the Digital Twin Challenge 2026
- **Reimagining & Reforming Healthcare in India Summit 2026**
- **ICMR**, **ADA**, and **RSSDI** for clinical guidelines referenced in risk scoring
- **Synthea** project for inspiring our synthetic EHR generation approach
- **OpenAPS Data Commons** for CGM data patterns

---

<div align="center">

**Built with ❤️ by Team Aethan — Amity University Mumbai**

*"The future of medicine is proactive, not reactive."*

</div>
