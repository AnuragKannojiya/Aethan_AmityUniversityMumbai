"""
Architecture Diagram Generator for GlucoTwin
=============================================
Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026

Generates publication-quality PDF and high-res PNG architecture diagrams.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = Path(__file__).resolve().parent

fig = plt.figure(figsize=(16, 10), dpi=300)
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Dark medical theme background
fig.patch.set_facecolor("#0a0e1a")
ax.set_facecolor("#0a0e1a")

# Title Banner
plt.text(8.0, 9.55, "GlucoTwin: Multimodal Digital Twin System Architecture", 
         ha="center", va="center", color="#38bdf8", fontsize=20, fontweight="bold", family="sans-serif")
plt.text(8.0, 9.20, "2-Hour Continuous Glucose Spike Prediction & In-Silico Intervention Simulation | Team Aethan, Amity University Mumbai", 
         ha="center", va="center", color="#94a3b8", fontsize=11, family="sans-serif")

# Helper for drawing rounded boxes
def draw_card(ax, x, y, w, h, bg_color, border_color, title, title_color="#38bdf8", subtitle=None):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=0.15",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.5,
        alpha=0.95
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.35, title, ha="center", va="center", 
            color=title_color, fontsize=12, fontweight="bold")
    if subtitle:
        ax.text(x + w/2, y + h - 0.65, subtitle, ha="center", va="center", 
                color="#64748b", fontsize=8.5, style="italic")

def draw_pill(ax, x, y, w, h, bg_color, text, text_color="#f8fafc", fontsize=9):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.08",
        facecolor=bg_color,
        edgecolor="#334155",
        linewidth=0.8
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", 
            color=text_color, fontsize=fontsize, fontweight="semibold")

def draw_arrow(ax, x1, y1, x2, y2, color="#06b6d4"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=15)
    )

# ----------------- SECTION 1: DATA SOURCES (Left) -----------------
draw_card(ax, 0.5, 4.8, 3.2, 3.8, "#111827", "#1e293b", "1. DATA INGESTION", "#38bdf8", "Dual Stream Telemetry")
draw_pill(ax, 0.7, 7.3, 2.8, 0.7, "#1e293b", "STATIC EHR STREAM (Synthea)\n• Demographics (Age, BMI, City)\n• Biomarkers (HbA1c, Fasting BG)\n• Meds, Comorbidities (46 feats)", fontsize=7.5)
draw_pill(ax, 0.7, 5.2, 2.8, 1.7, "#1e293b", "DYNAMIC WEARABLE/CGM\n• 5-min Interstitial Glucose\n• Heart Rate & SDNN HRV\n• Pedometer / Activity Mode\n• Sleep Stages & Duration\n• 1.73M Readings / 200 Pts", fontsize=7.5)

# ----------------- SECTION 2: MULTIMODAL FEATURE FUSION (Middle-Left) -----------------
draw_card(ax, 4.2, 4.8, 3.4, 3.8, "#111827", "#1e293b", "2. FEATURE PIPELINE", "#38bdf8", "FeatureStore & Leak-Free Splits")
draw_pill(ax, 4.4, 7.4, 3.0, 0.6, "#1e293b", "Static Preprocessing (46 feats)\nICMR Asian Indian BMI, TG/HDL, CKD", fontsize=7.5)
draw_pill(ax, 4.4, 6.4, 3.0, 0.8, "#1e293b", "Temporal Kinetics (54 feats)\n• 2h Rolling Mean, Std, Accel, CV%\n• Diurnal Circadian & Meal Clocks", fontsize=7.5)
draw_pill(ax, 4.4, 5.1, 3.0, 1.0, "#1e293b", "Patient-Stratified Splitter\nTrain (140) | Val (20) | Test (40)\nStrict zero-leakage sequence grid\nX_temp: (24x54), X_stat: (46)", fontsize=7.5)

# ----------------- SECTION 3: MULTIMODAL AI ENSEMBLE (Center-Right) -----------------
draw_card(ax, 8.1, 4.8, 3.8, 3.8, "#111827", "#0284c7", "3. HYBRID AI ENSEMBLE", "#38bdf8", "Calibrated Forecasting Engine")
draw_pill(ax, 8.3, 7.3, 3.4, 0.8, "#0f172a", "BRANCH A: PyTorch Attention-LSTM (0.6)\n• 2-Layer BiLSTM (128 units) + Self-Attn\n• Temporal saliency weighting across 2h window", fontsize=7.5, text_color="#38bdf8")
draw_pill(ax, 8.3, 6.3, 3.4, 0.7, "#0f172a", "BRANCH B: Tabular XGBoost (0.4)\n• 478 Rolling stats + static features\n• Asymmetric cost-sensitive scale_pos_weight", fontsize=7.5, text_color="#34d399")
draw_pill(ax, 8.3, 5.1, 3.4, 0.9, "#0369a1", "GLUCOTWIN ENSEMBLE FUSION\nROC-AUC: 0.993 | PR-AUC: 0.997 | Acc: 95.3%\n2-Hour Forward Spike Probability (>180 mg/dL)", fontsize=8, text_color="#ffffff")

# ----------------- SECTION 4: DIGITAL TWIN IN-SILICO ENGINE (Right) -----------------
draw_card(ax, 12.4, 4.8, 3.1, 3.8, "#111827", "#1e293b", "4. DIGITAL TWIN ENGINE", "#38bdf8", "In-Silico Virtual Patient")
draw_pill(ax, 12.6, 7.3, 2.7, 0.8, "#1e293b", "Physiological State Tracker\nLive Glycemic Velocity & Basal Targets", fontsize=7.5)
draw_pill(ax, 12.6, 6.2, 2.7, 0.8, "#1e293b", "Clinical Risk Calculator\nADA / RSSDI Tiering (Low->Critical)", fontsize=7.5)
draw_pill(ax, 12.6, 5.1, 2.7, 0.9, "#1e293b", "In-Silico What-If Simulator\nCounterfactual Trials:\n• GLUT4 Brisk Walk (-48 mg/dL)\n• Med Titration (-38 mg/dL)", fontsize=7.5)

# Connectors for Top Row
draw_arrow(ax, 3.7, 6.7, 4.2, 6.7)
draw_arrow(ax, 7.6, 6.7, 8.1, 6.7)
draw_arrow(ax, 11.9, 6.7, 12.4, 6.7)

# ----------------- SECTION 5: CLINICAL APPLICATION LAYER (Bottom Row) -----------------
draw_card(ax, 0.5, 0.6, 7.1, 3.6, "#111827", "#1e293b", "FASTAPI HIGH-PERFORMANCE BACKEND", "#38bdf8", "RESTful Clinical Microservice (0.0.0.0:8000)")
draw_pill(ax, 0.8, 3.0, 6.5, 0.45, "#1e293b", "GET /api/patients/:id/dashboard - Consolidated Clinical Payload in <20ms", fontsize=8)
draw_pill(ax, 0.8, 2.4, 6.5, 0.45, "#1e293b", "GET /api/patients/:id/predict - 2-Hour Trajectory, Spikes, Saliency Attention", fontsize=8)
draw_pill(ax, 0.8, 1.8, 6.5, 0.45, "#1e293b", "POST /api/patients/:id/simulate - In-Silico Intervention Counterfactuals", fontsize=8)
draw_pill(ax, 0.8, 1.2, 6.5, 0.45, "#1e293b", "GET /api/patients/:id/alerts - Acute Hypoglycemia & Critical Hyperglycemia Alerts", fontsize=8)
draw_pill(ax, 0.8, 0.7, 6.5, 0.4, "#0f766e", "DPDP Act 2023 & HIPAA Compliant Sandbox | 100% Synthetic Anonymized Telemetry", fontsize=8, text_color="#ccfbf1")

draw_card(ax, 8.1, 0.6, 7.4, 3.6, "#111827", "#059669", "NEXT.JS 14 CLINICAL TWIN DASHBOARD", "#34d399", "Real-Time Physician Telehealth Portal")
draw_pill(ax, 8.4, 3.0, 6.8, 0.45, "#1e293b", "PatientOverview - Demographics, Indian Clinical Phenotype, HbA1c, Meds", fontsize=8)
draw_pill(ax, 8.4, 2.4, 6.8, 0.45, "#1e293b", "GlucoseChart - Recharts 24h Trend + 2h Dashed Forecast with Euglycemic Bands", fontsize=8)
draw_pill(ax, 8.4, 1.8, 6.8, 0.45, "#1e293b", "RiskGauge & Timeline - Real-time Spike Probability, Time-to-Spike & Feature Drivers", fontsize=8)
draw_pill(ax, 8.4, 1.2, 6.8, 0.45, "#1e293b", "TwinVisualization - Multi-Organ Strain Meters & Interactive 'What-If' Modal", fontsize=8)
draw_pill(ax, 8.4, 0.7, 6.8, 0.4, "#134e4a", "Dark Medical Aesthetic | Responsive CSS Grid | 30s Real-time Live Polling", fontsize=8, text_color="#a7f3d0")

# Connectors to Bottom Row
draw_arrow(ax, 13.9, 4.8, 13.9, 4.2)
draw_arrow(ax, 13.9, 4.2, 4.0, 4.2)
draw_arrow(ax, 4.0, 4.2, 4.0, 3.8)
draw_arrow(ax, 7.6, 2.4, 8.1, 2.4)

plt.tight_layout()
pdf_path = OUTPUT_DIR / "architecture_diagram.pdf"
png_path = OUTPUT_DIR / "architecture_diagram.png"
fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)
plt.close(fig)

print(f"Architecture diagrams generated successfully:\n - {pdf_path}\n - {png_path}")
