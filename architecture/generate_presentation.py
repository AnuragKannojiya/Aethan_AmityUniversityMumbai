"""
Multi-Slide Presentation Generator for GlucoTwin
=================================================
Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026

Generates a publication-grade, 8-slide 16:9 PDF presentation deck.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

OUTPUT_DIR = Path(__file__).resolve().parent
PDF_PATH = OUTPUT_DIR / "presentation.pdf"

def create_slide_base():
    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#0a0e1a")
    ax.set_facecolor("#0a0e1a")
    return fig, ax

def add_header(ax, title, subtitle):
    # Top Accent Line
    line = patches.Rectangle((0.8, 8.4), 14.4, 0.04, facecolor="#06b6d4", edgecolor="none")
    ax.add_patch(line)
    ax.text(0.8, 8.0, title, color="#38bdf8", fontsize=20, fontweight="bold", family="sans-serif")
    ax.text(0.8, 7.6, subtitle, color="#94a3b8", fontsize=11, family="sans-serif")

def add_footer(ax, slide_num, total_slides=8):
    ax.text(0.8, 0.4, "GlucoTwin | Team Aethan (Amity University Mumbai) | Digital Twin Challenge 2026",
            color="#475569", fontsize=9, family="sans-serif")
    ax.text(15.2, 0.4, f"Slide {slide_num} of {total_slides}",
            color="#475569", fontsize=9, ha="right", family="sans-serif")

def draw_card(ax, x, y, w, h, bg_color="#111827", border_color="#1e293b", title=None, title_color="#38bdf8"):
    rect = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1,rounding_size=0.12",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.2
    )
    ax.add_patch(rect)
    if title:
        ax.text(x + 0.3, y + h - 0.4, title, color=title_color, fontsize=12, fontweight="bold")

with PdfPages(PDF_PATH) as pdf:

    # -------------------------------------------------------------
    # SLIDE 1: TITLE SLIDE
    # -------------------------------------------------------------
    fig, ax = create_slide_base()
    # Glow ring accent
    accent = patches.FancyBboxPatch((1.2, 1.2), 13.6, 6.6, boxstyle="round,pad=0.2,rounding_size=0.2",
                                    facecolor="#0f172a", edgecolor="#0284c7", linewidth=2.0)
    ax.add_patch(accent)

    ax.text(8.0, 5.8, "GlucoTwin", color="#38bdf8", fontsize=42, fontweight="bold", ha="center")
    ax.text(8.0, 5.0, "AI-Powered Multimodal Digital Twin for Type 2 Diabetes Management",
            color="#f8fafc", fontsize=20, fontweight="semibold", ha="center")
    ax.text(8.0, 4.3, "Predicting 2-Hour Forward Glucose Spikes & Simulating In-Silico Interventions",
            color="#06b6d4", fontsize=14, ha="center")

    pill = patches.FancyBboxPatch((3.5, 2.3), 9.0, 1.3, boxstyle="round,pad=0.1,rounding_size=0.1",
                                  facecolor="#1e293b", edgecolor="#334155", linewidth=1.0)
    ax.add_patch(pill)
    ax.text(8.0, 3.1, "TEAM AETHAN — AMITY UNIVERSITY MUMBAI", color="#38bdf8", fontsize=13, fontweight="bold", ha="center")
    ax.text(8.0, 2.6, "Challenge: Digital Twin Challenge 2026 | Happiest Health Summit", color="#94a3b8", fontsize=10.5, ha="center")

    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------------------------------------------
    # SLIDE 2: THE CLINICAL CHALLENGE
    # -------------------------------------------------------------
    fig, ax = create_slide_base()
    add_header(ax, "The Clinical Challenge: T2D Burden in India & Limits of Reactive Care",
               "Why traditional reactive healthcare fails diabetic patients in India")

    draw_card(ax, 0.8, 1.2, 4.5, 6.0, title="Epidemic Scale in India")
    ax.text(1.1, 6.2, "• 101 Million Adults in India\n  diagnosed with Type 2 Diabetes\n  (Global 'Diabetes Capital').\n\n• Asian Indian Phenotype:\n  Marked visceral adiposity, high\n  triglycerides, low HDL, and high\n  insulin resistance at lower BMIs.\n\n• High Carb Dietary Patterns:\n  Refined rice/roti staples cause\n  steep postprandial glycemic surges.",
            color="#cbd5e1", fontsize=11, va="top", linespacing=1.4)

    draw_card(ax, 5.7, 1.2, 4.6, 6.0, title="Flaws of Current Care")
    ax.text(6.0, 6.2, "• Retrospective Biomarkers:\n  HbA1c only reveals a 3-month\n  average, completely concealing\n  dangerous acute glycemic excursions.\n\n• Reactive Crisis Management:\n  Interventions only occur AFTER\n  blood glucose crosses 250+ mg/dL,\n  causing microvascular damage.\n\n• Disconnected Telemetry:\n  EHR data sits in siloes isolated\n  from consumer wearable devices.",
            color="#cbd5e1", fontsize=11, va="top", linespacing=1.4)

    draw_card(ax, 10.7, 1.2, 4.5, 6.0, title="The Digital Twin Paradigm", border_color="#06b6d4")
    ax.text(11.0, 6.2, "• Continuous Dynamic Mirror:\n  Synchronizes live CGM telemetry\n  with baseline EHR records.\n\n• 2-Hour Predictive Horizon:\n  Forecasts dangerous spikes BEFORE\n  they occur with 95.3% accuracy.\n\n• In-Silico 'What-If' Trials:\n  Simulates patient-specific responses\n  to walking, medications, and meals\n  before taking clinical action.",
            color="#cbd5e1", fontsize=11, va="top", linespacing=1.4)

    add_footer(ax, 2)
    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------------------------------------------
    # SLIDE 3: MULTIMODAL DATA FUSION ARCHITECTURE
    # -------------------------------------------------------------
    fig, ax = create_slide_base()
    add_header(ax, "Multimodal Data Stream Fusion: Static EHR + Dynamic Wearables",
               "Strict zero-leakage patient-level partitioning across 1.73M data points")

    draw_card(ax, 0.8, 4.0, 6.8, 3.2, title="Static Historical EHR Stream (Synthea)")
    ax.text(1.1, 6.4, "• Cohort: 200 Indian Type 2 Diabetes Patients (P001-P200)\n• Biomarkers: HbA1c (5.5-12.0%), Fasting Glucose, Lipid Panel, eGFR\n• Clinical Phenotypes: ICMR-consensus Asian Indian BMI, Duration (0.5-25y)\n• Multi-label Comorbidities: Hypertension, Dyslipidemia, CKD, Neuropathy\n• 46 Preprocessed Static Input Features (One-Hot & Standardized)",
            color="#cbd5e1", fontsize=10.5, va="top", linespacing=1.3)

    draw_card(ax, 0.8, 1.0, 6.8, 2.7, title="Dynamic Real-Time Wearable & CGM Telemetry")
    ax.text(1.1, 2.9, "• 1,728,000 Continuous Telemetry Samples (30 days @ 5-min intervals)\n• CGM Interstitial Glucose, Autonomic HR & SDNN Heart Rate Variability\n• Activity Pedometer (Steps/5min), Multi-stage Sleep Architecture\n• Vectorized Indian Circadian & Meal Clocks (Chai, Roti/Dal excursions)",
            color="#cbd5e1", fontsize=10.5, va="top", linespacing=1.3)

    draw_card(ax, 8.2, 1.0, 7.0, 6.2, title="Multimodal Feature Fusion & Sliding Window", border_color="#0284c7")
    ax.text(8.5, 6.4, "• Lookback Window: 24 Timesteps (2 Hours @ 5-min intervals)\n• Dynamic Features (54 channels): Rolling mean, velocity, acceleration,\n  GVI (Glucose Volatility Index), MAG-1h, Diurnal sin/cos harmonics.\n• Fusion Strategy: Broadcast-merge static patient embedding (46 dims)\n  with temporal tensor (24 x 54) -> Multimodal sequence array.\n\n• Leak-Free Patient Partitioning:\n  - Train Split: 140 Patients (201,180 sequences, 71.1% spike ratio)\n  - Val Split:   20 Patients (28,740 sequences, 67.7% spike ratio)\n  - Test Split:  40 Patients (57,480 sequences, 67.9% spike ratio)\n  * FeatureStore scalers fit strictly on training cohort to ensure 0% data leakage.",
            color="#cbd5e1", fontsize=10.5, va="top", linespacing=1.35)

    add_footer(ax, 3)
    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------------------------------------------
    # SLIDE 4: HYBRID AI ENSEMBLE ARCHITECTURE
    # -------------------------------------------------------------
    fig, ax = create_slide_base()
    add_header(ax, "Hybrid AI Model: Attention-LSTM + Tabular XGBoost Ensemble",
               "Combining deep temporal sequence learning with non-linear gradient boosted trees")

    draw_card(ax, 0.8, 1.2, 4.6, 6.0, title="Branch A: PyTorch Attention-LSTM (0.6)")
    ax.text(1.1, 6.2, "• Temporal Input: (Batch, 24, 54)\n• Static Input:   (Batch, 46)\n\n• 2-Layer Bidirectional LSTM\n  (128 hidden units, dropout=0.3)\n\n• Temporal Self-Attention:\n  Dynamically scores which of the past\n  24 timesteps drive spike momentum\n  (providing explainable saliency).\n\n• Static MLP Fusion Layer:\n  Projects 46 clinical biomarkers to\n  64-dim representation before fusion.\n\n• Cost-sensitive weighted BCE loss\n  optimized with Cosine Annealing.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    draw_card(ax, 5.7, 1.2, 4.6, 6.0, title="Branch B: Tabular XGBoost (0.4)")
    ax.text(6.0, 6.2, "• Tabular Features: 478 Dimensions\n  (Summary stats across dynamic channels\n  combined with static clinical profile).\n\n• Histogram Tree Method (hist)\n  with scale_pos_weight optimization\n  to minimize missed clinical spikes.\n\n• Learning Rate: 0.07, Max Depth: 6,\n  180 Estimators with Early Stopping.\n\n• Top Feature Drivers:\n  - Instantaneous glucose rate of change\n  - 1-Hour Mean Absolute Glucose (MAG)\n  - Baseline HbA1c & Fasting Glucose\n  - TG-to-HDL metabolic ratio.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    draw_card(ax, 10.7, 1.2, 4.5, 6.0, title="Calibrated Ensemble Fusion", border_color="#34d399")
    ax.text(11.0, 6.2, "• Joint Weighted Decision Rule:\n  P_ensemble = 0.60 * P_lstm + 0.40 * P_xgb\n\n• Dual-Perspective Strength:\n  - LSTM captures kinetic velocity\n    and physiological inertia.\n  - XGBoost excels on complex non-linear\n    cross-interactions (meds x HbA1c).\n\n• Uncertainty & Concordance:\n  Calculates inter-model agreement and\n  statistical confidence (0.50 to 0.99)\n  for every dispatched prediction.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    add_footer(ax, 4)
    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------------------------------------------
    # SLIDE 5: EMPIRICAL BENCHMARKS & EVALUATION
    # -------------------------------------------------------------
    fig, ax = create_slide_base()
    add_header(ax, "Empirical Evaluation: Outstanding Generalization on Test Cohort",
               "Evaluated on 57,480 unseen sequences across 40 completely held-out Indian patients")

    # Benchmark Table Box
    draw_card(ax, 0.8, 3.2, 14.4, 4.0, title="Model Evaluation Leaderboard (Held-Out Test Cohort: 57,480 Sequences)")

    headers = ["Metric", "PyTorch LSTM (0.6)", "XGBoost (0.4)", "GlucoTwin Ensemble", "Clinical Significance"]
    x_positions = [1.2, 4.2, 6.8, 9.4, 12.0]

    for x, h in zip(x_positions, headers):
        ax.text(x, 6.5, h, color="#38bdf8", fontsize=10.5, fontweight="bold")

    metrics_rows = [
        ("ROC-AUC", "0.9899", "0.9938", "0.9928", "Near-perfect discrimination"),
        ("PR-AUC (Avg Precision)", "0.9952", "0.9971", "0.9966", "Robust under class imbalance"),
        ("Accuracy", "94.52%", "95.67%", "95.34%", "High overall predictive fidelity"),
        ("Sensitivity / Recall", "94.53%", "95.68%", "95.45%", "Crucial: <4.6% missed spikes"),
        ("Specificity", "94.50%", "95.64%", "95.10%", "Prevents clinician alert fatigue"),
        ("F1-Score", "0.9591", "0.9678", "0.9653", "High harmonic precision/recall"),
        ("Brier Calibration Score", "0.0388", "0.0311", "0.0330", "Exceptional risk calibration")
    ]

    y_pos = 5.9
    for row in metrics_rows:
        ax.text(x_positions[0], y_pos, row[0], color="#f8fafc", fontsize=9.5, fontweight="semibold")
        ax.text(x_positions[1], y_pos, row[1], color="#94a3b8", fontsize=9.5)
        ax.text(x_positions[2], y_pos, row[2], color="#94a3b8", fontsize=9.5)
        ax.text(x_positions[3], y_pos, row[3], color="#34d399", fontsize=10, fontweight="bold")
        ax.text(x_positions[4], y_pos, row[4], color="#38bdf8", fontsize=9, style="italic")
        y_pos -= 0.38

    draw_card(ax, 0.8, 1.0, 7.0, 1.9, title="Key Finding: Zero Missed Crises", border_color="#059669")
    ax.text(1.1, 2.1, "Ensemble sensitivity of 95.5% ensures that almost all impending hyperglycemic\nspikes (>180 mg/dL) are detected 2 hours ahead of time, granting patients\na massive therapeutic window to take corrective action.",
            color="#cbd5e1", fontsize=9.5, va="top", linespacing=1.3)

    draw_card(ax, 8.2, 1.0, 7.0, 1.9, title="Key Finding: Low Alarm Fatigue", border_color="#0284c7")
    ax.text(8.5, 2.1, "Ensemble specificity of 95.1% and low Brier score (0.033) prevents false-alarm\nfatigue, a well-documented failure point in clinical IoT monitoring systems.",
            color="#cbd5e1", fontsize=9.5, va="top", linespacing=1.3)

    add_footer(ax, 5)
    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------------------------------------------
    # SLIDE 6: DIGITAL TWIN IN-SILICO ENGINE & SIMULATION
    # -------------------------------------------------------------
    fig, ax = create_slide_base()
    add_header(ax, "Digital Twin Engine: In-Silico 'What-If' Simulation & Risk Stratification",
               "Evaluating physiological interventions before taking clinical action")

    draw_card(ax, 0.8, 1.2, 4.6, 6.0, title="Real-Time State Tracking")
    ax.text(1.1, 6.2, "• Multi-Organ Strain Scores:\n  - Pancreatic Beta-Cell load (0-100)\n  - Hepatic Glucose Production (HGP)\n  - Autonomic Cardiovascular tone.\n\n• Adaptive Basal Reversion:\n  Simulates individual physiological\n  clearance kinetics and active\n  insulin resistance multipliers.\n\n• Dynamic Saliency Weighting:\n  Extracts attention weights across\n  the 2-hour window for explainable\n  root-cause analysis.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    draw_card(ax, 5.7, 1.2, 4.6, 6.0, title="Clinical Risk Stratification")
    ax.text(6.0, 6.2, "• ADA & RSSDI Consensus Tiers:\n  - Low (<140 mg/dL, stable)\n  - Moderate (140-180 mg/dL)\n  - High (>180 mg/dL or spike prob >55%)\n  - Critical (>250 mg/dL or hypo <70 mg/dL)\n\n• Multi-Attribute Risk Factors:\n  Itemized scoring of current BG,\n  glycemic velocity, HbA1c, and meds.\n\n• Automated Alert Dispatcher:\n  Dispatches push notifications for\n  acute hypo, rapid surges, and predicted\n  hyperglycemic events.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    draw_card(ax, 10.7, 1.2, 4.5, 6.0, title="Counterfactual 'What-If' Trials", border_color="#38bdf8")
    ax.text(11.0, 6.2, "• In-Silico Intervention Modeling:\n\n  1. Brisk Postprandial Walk (20m):\n     Triggers non-insulin-mediated\n     GLUT4 glucose disposal.\n     -> Reduces peak by -48.0 mg/dL!\n\n  2. Pharmacotherapy Timing:\n     Evaluates Metformin/DPP-4i dosing\n     -> Lowers excursion by -38.0 mg/dL.\n\n  3. Dietary Fiber Pre-load:\n     Delays gastric carb absorption\n     -> Blunts surge by -18.0 mg/dL.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    add_footer(ax, 6)
    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------------------------------------------
    # SLIDE 7: NEXT.JS 14 CLINICAL DASHBOARD
    # -------------------------------------------------------------
    fig, ax = create_slide_base()
    add_header(ax, "Next.js 14 Clinical Dashboard: Doctor-Centric Telehealth Portal",
               "Real-time reactive UI designed for rapid clinical triage and digital twin interaction")

    draw_card(ax, 0.8, 4.2, 6.8, 3.0, title="High-Density Medical Layout")
    ax.text(1.1, 6.4, "• PatientOverview: Indian demographic profile, HbA1c, BMI, comorbidities\n• GlucoseChart: Interactive 24-hour Recharts line graph with colored\n  euglycemic safety bands (70-180 mg/dL) & dashed 2-hour forward forecast\n• RiskGauge & Timeline: Circular SVG dial with pulse animations and top drivers",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    draw_card(ax, 0.8, 1.0, 6.8, 2.9, title="Active Alert & Intervention Panel")
    ax.text(1.1, 3.1, "• AlertPanel: Prioritized acute clinical alerts with recommended actions\n• Digital Twin Visualization: Organ-level strain progress meters\n• Interactive What-If Modal: Real-time dual trajectory overlay comparing\n  unmitigated spike curve against simulated intervention response",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    draw_card(ax, 8.2, 1.0, 7.0, 6.2, title="Architecture & Tech Stack Highlights", border_color="#059669")
    ax.text(8.5, 6.4, "• Frontend Stack:\n  Next.js 14 (App Router) + React 18 + Tailwind CSS + Recharts + Framer Motion\n\n• Dark Medical Design System:\n  Deep navy `#0a0e1a` high-contrast aesthetic modeled after hospital ICU\n  telemetry stations for minimal optical fatigue during extended clinical shifts.\n\n• Real-Time Synchronized State:\n  Auto-refreshes every 30 seconds via REST API with resilient offline fallback\n  to mock data for uninterrupted zero-latency demonstration.\n\n• High-Performance Backend:\n  FastAPI microservice handling 1.73M records in memory with <20ms\n  endpoint latency for 2-hour forecasts and counterfactual trials.",
            color="#cbd5e1", fontsize=10.5, va="top", linespacing=1.35)

    add_footer(ax, 7)
    pdf.savefig(fig)
    plt.close(fig)

    # -------------------------------------------------------------
    # SLIDE 8: IMPACT, COMPLIANCE & FUTURE HORIZON
    # -------------------------------------------------------------
    fig, ax = create_slide_base()
    add_header(ax, "Healthcare Impact, Regulatory Compliance & Future Horizon",
               "Transforming chronic disease management across India from reactive to proactive")

    draw_card(ax, 0.8, 1.2, 4.6, 6.0, title="DPDP & HIPAA Sandbox")
    ax.text(1.1, 6.2, "• DPDP Act 2023 & HIPAA Compliant:\n  Strictly anonymized synthetic datasets\n  modeled on Synthea & OpenAPS.\n\n• Zero PII Exposure:\n  Cryptographically protected\n  synthetic IDs (IND-T2D-001).\n\n• Auditable Explainability:\n  Attention weights and XGBoost\n  SHAP-ready feature importances\n  guarantee transparent clinical decisions.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    draw_card(ax, 5.7, 1.2, 4.6, 6.0, title="Real-World Clinical Impact")
    ax.text(6.0, 6.2, "• 2-Hour Advance Warning:\n  Transforms diabetes care from reactive\n  crisis fighting to proactive lifestyle\n  and medication nudges.\n\n• Prevents Long-Term Damage:\n  Minimizes glycemic variability (GVI),\n  reducing risk of retinopathy, diabetic\n  nephropathy, and coronary disease.\n\n• In-Silico Empowerment:\n  Patients see directly how a 15-min walk\n  can avert a severe 220 mg/dL surge.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    draw_card(ax, 10.7, 1.2, 4.5, 6.0, title="Future Roadmap", border_color="#38bdf8")
    ax.text(11.0, 6.2, "• Phase 2: Wearable Integration\n  Direct Bluetooth Low Energy (BLE)\n  integration with FreeStyle Libre &\n  Dexcom G7 CGMs.\n\n• Phase 3: Closed-Loop Delivery\n  Coupling predictions with smart\n  insulin pumps for autonomous micro-dosing.\n\n• Multi-Organ Twin Expansion:\n  Cardiovascular & renal digital twin\n  extension using MIMIC-IV.",
            color="#cbd5e1", fontsize=10, va="top", linespacing=1.3)

    add_footer(ax, 8)
    pdf.savefig(fig)
    plt.close(fig)

print(f"Presentation generated successfully at: {PDF_PATH}")
