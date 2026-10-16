"""
Jupyter Notebook Generator for GlucoTwin
=========================================
Generates 3 interactive Jupyter notebooks in notebooks/
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent

def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            },
            "orig_nbformat": 4
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.split("\n")]
    }

# -----------------------------------------------------------------------------
# 1. NOTEBOOK 01: DATA EXPLORATION
# -----------------------------------------------------------------------------
nb1_cells = [
    markdown_cell("""# GlucoTwin: 01. Exploratory Data Analysis
**Team:** Aethan, Amity University Mumbai  
**Challenge:** Digital Twin Challenge 2026 — Happiest Health Summit  

This notebook explores the raw multi-modal datasets:
1. Static Electronic Health Records (EHR) of 200 Indian Type 2 Diabetes patients.
2. Dynamic continuous wearable and CGM telemetry (1.73M 5-minute readings).
3. Postprandial glucose dynamics, dawn phenomenon, and Indian dietary patterns.
"""),
    code_cell("""import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Project root:", PROJECT_ROOT)
"""),
    markdown_cell("## 1. Load & Inspect Static EHR Cohort"),
    code_cell("""ehr_path = PROJECT_ROOT / "data" / "raw" / "ehr_patients.csv"
ehr_df = pd.read_csv(ehr_path)
print(f"Total Enrolled Patients: {len(ehr_df)}")
ehr_df.head()
"""),
    code_cell("""# Summary statistics for clinical biomarkers
biomarkers = ['age', 'bmi', 'hba1c', 'fasting_glucose', 'blood_pressure_systolic', 'cholesterol_total']
ehr_df[biomarkers].describe().T
"""),
    code_cell("""# Distribution of HbA1c and Asian Indian BMI
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(ehr_df['hba1c'], kde=True, ax=axes[0], color='#0284c7', bins=20)
axes[0].axvline(7.0, color='green', linestyle='--', label='Target Goal (<7.0%)')
axes[0].axvline(9.0, color='red', linestyle='--', label='High Risk (>9.0%)')
axes[0].set_title("HbA1c (%) Distribution in Indian T2D Cohort", fontsize=12, fontweight='bold')
axes[0].legend()

sns.histplot(ehr_df['bmi'], kde=True, ax=axes[1], color='#0d9488', bins=20)
axes[1].axvline(23.0, color='orange', linestyle='--', label='Asian Indian Overweight (>23.0)')
axes[1].axvline(25.0, color='red', linestyle='--', label='Asian Indian Obese (>25.0)')
axes[1].set_title("Body Mass Index (BMI) Distribution (ICMR Guidelines)", fontsize=12, fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.show()
"""),
    markdown_cell("## 2. Load & Inspect Dynamic CGM Telemetry"),
    code_cell("""# Read first 10,000 readings of continuous CGM telemetry
cgm_path = PROJECT_ROOT / "data" / "raw" / "cgm_readings.csv"
cgm_sample = pd.read_csv(cgm_path, nrows=10000)
cgm_sample['timestamp'] = pd.to_datetime(cgm_sample['timestamp'])
print("CGM Telemetry Columns:", list(cgm_sample.columns))
cgm_sample.head()
"""),
    code_cell("""# Plot 48-Hour Interstitial Glucose Trajectory for Patient P001
p1_readings = cgm_sample[cgm_sample['patient_id'] == 'P001'].head(288 * 2) # 48 hours

plt.figure(figsize=(15, 6))
plt.plot(p1_readings['timestamp'], p1_readings['glucose_mg_dl'], color='#0284c7', lw=1.8, label='CGM Interstitial Glucose')
plt.axhspan(70, 180, color='green', alpha=0.15, label='Target Euglycemic Zone (70-180 mg/dL)')
plt.axhline(180, color='orange', linestyle='--', alpha=0.7, label='Spike Threshold (180 mg/dL)')
plt.axhline(250, color='red', linestyle='--', alpha=0.7, label='Critical Hyperglycemia (>250 mg/dL)')

plt.title("48-Hour Continuous Glucose Telemetry — Patient P001 (Mumbai, 52y, HbA1c 8.7%)", fontsize=14, fontweight='bold')
plt.xlabel("Timeline", fontsize=11)
plt.ylabel("Glucose (mg/dL)", fontsize=11)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
""")
]

with open(NOTEBOOKS_DIR / "01_data_exploration.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb1_cells), f, indent=2)

# -----------------------------------------------------------------------------
# 2. NOTEBOOK 02: FEATURE ENGINEERING
# -----------------------------------------------------------------------------
nb2_cells = [
    markdown_cell("""# GlucoTwin: 02. Feature Engineering & Multimodal Fusion Pipeline
**Team:** Aethan, Amity University Mumbai  
**Challenge:** Digital Twin Challenge 2026 — Happiest Health Summit  

This notebook demonstrates:
1. Static EHR feature extraction (46 clinical features).
2. Dynamic temporal feature extraction (54 channels: rolling means, momentum, circadian harmonics).
3. Leak-free sliding window sequence creation for PyTorch LSTM and XGBoost models.
"""),
    code_cell("""import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering.static_features import extract_static_features
from src.feature_engineering.temporal_features import extract_temporal_features
from src.feature_engineering.feature_store import FeatureStore
"""),
    markdown_cell("## 1. Inspect Feature Store Metadata"),
    code_cell("""meta_path = PROJECT_ROOT / "data" / "processed" / "feature_metadata.json"
with open(meta_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Feature Summary:")
for k, v in metadata['summary'].items():
    print(f"  {k}: {v}")
"""),
    markdown_cell("## 2. Load Processed Multimodal Dataset Splits (.npz)"),
    code_cell("""train_data = np.load(PROJECT_ROOT / "data" / "processed" / "train_data.npz")
val_data = np.load(PROJECT_ROOT / "data" / "processed" / "val_data.npz")
test_data = np.load(PROJECT_ROOT / "data" / "processed" / "test_data.npz")

print("Train Split:")
print("  X_temporal shape:", train_data['X_temporal'].shape)
print("  X_static shape:  ", train_data['X_static'].shape)
print("  y shape:         ", train_data['y'].shape, f"(Spike ratio: {train_data['y'].mean()*100:.1f}%)")

print("\\nTest Split (Held-Out):")
print("  X_temporal shape:", test_data['X_temporal'].shape)
print("  X_static shape:  ", test_data['X_static'].shape)
print("  y shape:         ", test_data['y'].shape, f"(Spike ratio: {test_data['y'].mean()*100:.1f}%)")
""")
]

with open(NOTEBOOKS_DIR / "02_feature_engineering.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb2_cells), f, indent=2)

# -----------------------------------------------------------------------------
# 3. NOTEBOOK 03: MODEL TRAINING & EVALUATION
# -----------------------------------------------------------------------------
nb3_cells = [
    markdown_cell("""# GlucoTwin: 03. Model Training, Ensemble Evaluation & In-Silico Twin
**Team:** Aethan, Amity University Mumbai  
**Challenge:** Digital Twin Challenge 2026 — Happiest Health Summit  

This notebook evaluates:
1. PyTorch Attention-LSTM + XGBoost ensemble performance on 57,480 held-out test sequences.
2. Clinical confusion matrix, ROC-AUC, PR-AUC, and feature importance rankings.
3. Patient-specific Digital Twin state synchronization and 'What-If' intervention simulation.
"""),
    code_cell("""import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

PROJECT_ROOT = Path.cwd().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble import GlucoTwinEnsemble
from src.digital_twin.twin_engine import DigitalTwin
"""),
    markdown_cell("## 1. Load Trained GlucoTwin Ensemble"),
    code_cell("""models_dir = PROJECT_ROOT / "data" / "processed" / "models"
ensemble = GlucoTwinEnsemble.load(models_dir)
print("Ensemble successfully loaded!")
print("  LSTM weights:", ensemble.lstm_weight)
print("  XGBoost weights:", ensemble.xgb_weight)
"""),
    markdown_cell("## 2. Benchmark Metrics Leaderboard"),
    code_cell("""metrics_path = models_dir / "evaluation_metrics.json"
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# Display tabular comparison
comparison = pd.DataFrame({
    'Metric': ['ROC-AUC', 'PR-AUC (Avg Prec)', 'Accuracy', 'Sensitivity (Recall)', 'Specificity', 'F1-Score', 'Brier Score'],
    'LSTM (0.6)': [
        metrics['lstm_model']['roc_auc'],
        metrics['lstm_model']['pr_auc'],
        f"{metrics['lstm_model']['accuracy']*100:.2f}%",
        f"{metrics['lstm_model']['recall_sensitivity']*100:.2f}%",
        f"{metrics['lstm_model']['specificity']*100:.2f}%",
        metrics['lstm_model']['f1_score'],
        metrics['lstm_model']['brier_score']
    ],
    'XGBoost (0.4)': [
        metrics['xgboost_model']['roc_auc'],
        metrics['xgboost_model']['pr_auc'],
        f"{metrics['xgboost_model']['accuracy']*100:.2f}%",
        f"{metrics['xgboost_model']['recall_sensitivity']*100:.2f}%",
        f"{metrics['xgboost_model']['specificity']*100:.2f}%",
        metrics['xgboost_model']['f1_score'],
        metrics['xgboost_model']['brier_score']
    ],
    'GlucoTwin Ensemble': [
        metrics['glucotwin_ensemble']['roc_auc'],
        metrics['glucotwin_ensemble']['pr_auc'],
        f"{metrics['glucotwin_ensemble']['accuracy']*100:.2f}%",
        f"{metrics['glucotwin_ensemble']['recall_sensitivity']*100:.2f}%",
        f"{metrics['glucotwin_ensemble']['specificity']*100:.2f}%",
        metrics['glucotwin_ensemble']['f1_score'],
        metrics['glucotwin_ensemble']['brier_score']
    ]
})

comparison
"""),
    markdown_cell("## 3. Top Feature Importance (XGBoost Branch)"),
    code_cell("""fi_path = models_dir / "feature_importance.json"
with open(fi_path, "r", encoding="utf-8") as f:
    fi_data = json.load(f)

fi_df = pd.DataFrame(fi_data).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df, x='importance', y='feature', palette='Blues_r')
plt.title("Top 15 Predictive Features for 2-Hour Glucose Spike", fontsize=13, fontweight='bold')
plt.xlabel("Importance Weight", fontsize=11)
plt.ylabel("Engineered Feature", fontsize=11)
plt.tight_layout()
plt.show()
"""),
    markdown_cell("## 4. In-Silico Digital Twin 'What-If' Simulation"),
    code_cell("""# Instantiate Digital Twin for Patient P001
patient_profile = {
    "patient_id": "P001",
    "name": "Rajesh Sharma",
    "age": 52,
    "gender": "M",
    "city": "Mumbai",
    "bmi": 28.4,
    "hba1c": 8.7,
    "fasting_glucose": 158.0,
    "diabetes_duration_years": 6.5,
    "medication": "Metformin + Sitagliptin",
    "comorbidities": "Hypertension, Dyslipidemia"
}

twin = DigitalTwin(patient_id="P001", ehr_data=patient_profile, model_ensemble=ensemble)

# Ingest recent streaming telemetry showing upward glucose surge
readings = [
    {"timestamp": "2026-10-15T12:00:00Z", "glucose": 155.0, "heart_rate": 74, "steps": 50, "activity_type": "Sedentary"},
    {"timestamp": "2026-10-15T12:05:00Z", "glucose": 168.0, "heart_rate": 78, "steps": 60, "activity_type": "Sedentary"},
    {"timestamp": "2026-10-15T12:10:00Z", "glucose": 182.0, "heart_rate": 80, "steps": 40, "activity_type": "Sedentary"}
]
twin.update_state(readings)

# Run counterfactual trial: What if patient takes a 20-min brisk walk?
sim_walk = twin.simulate_intervention(intervention_type="walk", intensity="moderate")

print("Expected Clinical Improvement:")
print(sim_walk['expected_improvement'])
print(f"Peak Reduction: {sim_walk['peak_reduction_mg_dl']} mg/dL")
print(f"Spike Averted: {sim_walk['spike_averted']}")
"""),
    code_cell("""# Plot Counterfactual In-Silico Trajectory
base_vals = [p['glucose'] for p in sim_walk['baseline_trajectory']]
walk_vals = [p['glucose'] for p in sim_walk['intervention_trajectory']]
time_steps = [f"+{i*10}m" for i in range(1, len(base_vals) + 1)]

plt.figure(figsize=(12, 5))
plt.plot(time_steps, base_vals, 'r--', lw=2.2, label='Baseline (No Intervention - Spike to >215 mg/dL)')
plt.plot(time_steps, walk_vals, 'g-', lw=2.5, label='Counterfactual: 20-min Brisk Walk (GLUT4 Disposal)')
plt.axhspan(70, 180, color='green', alpha=0.15, label='Euglycemic Safe Zone (70-180 mg/dL)')
plt.axhline(180, color='red', linestyle=':', alpha=0.8, label='Spike Threshold (180 mg/dL)')

plt.title("In-Silico Digital Twin 'What-If' Simulation — Patient P001", fontsize=13, fontweight='bold')
plt.xlabel("Forecast Horizon (Minutes)", fontsize=11)
plt.ylabel("Glucose (mg/dL)", fontsize=11)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
""")
]

with open(NOTEBOOKS_DIR / "03_model_training_and_evaluation.ipynb", "w", encoding="utf-8") as f:
    json.dump(make_notebook(nb3_cells), f, indent=2)

print("Notebooks created successfully in:", NOTEBOOKS_DIR)
