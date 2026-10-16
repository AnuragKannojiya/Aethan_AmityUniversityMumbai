# GlucoTwin: 20-Minute Jury Presentation & Demo Script

**Team:** Aethan  
**Institution:** Amity University Mumbai  
**Challenge:** Digital Twin Challenge 2026 — Happiest Health Summit  
**Project:** GlucoTwin: AI-Powered Multimodal Digital Twin for 2-Hour Glucose Spike Prediction in Type 2 Diabetes  

---

## 🎙️ Video Recording Guide & Time Allocation (Total: 20 Minutes)

| Segment | Duration | Speaker | Focus |
|---|---|---|---|
| **Part 1: Introduction & Healthcare Context** | 3 min (0:00 - 3:00) | Speaker 1 | Problem statement, Indian diabetic epidemic, limitations of reactive HbA1c |
| **Part 2: Dual Data Stream Architecture** | 4 min (3:00 - 7:00) | Speaker 2 | Synthea EHR (46 feats) + CGM/wearables (1.73M readings), feature engineering & zero-leakage splits |
| **Part 3: Hybrid AI Ensemble & Benchmarks** | 4 min (7:00 - 11:00) | Speaker 1 | PyTorch Attention-LSTM + XGBoost ensemble, ROC-AUC 0.993, PR-AUC 0.997, 95.5% sensitivity |
| **Part 4: Digital Twin Engine & 'What-If' Simulation** | 3 min (11:00 - 14:00) | Speaker 2 | In-silico organ strain modeling, GLUT4 walk simulation (-48 mg/dL), counterfactuals |
| **Part 5: Live Prototype Demonstration** | 4 min (14:00 - 18:00) | Both | Next.js 14 dashboard walk-through, live FastAPI backend, real-time alert triage, what-if modal |
| **Part 6: DPDP/HIPAA Compliance & Conclusion** | 2 min (18:00 - 20:00) | Speaker 1 | DPDP Act 2023 compliance, clinical impact, closed-loop future horizon |

---

## 📜 Full Word-for-Word Presentation Script

### Part 1: Introduction & Healthcare Problem Statement (0:00 - 3:00)
**[Show Slide 1: Title Slide & Slide 2: The Clinical Challenge]**

**Speaker 1:**
> "Respected jury members, clinical experts, and attendees of the Happiest Health Summit 2026. We are **Team Aethan** from **Amity University Mumbai**, and today we are honored to introduce **GlucoTwin** — an AI-powered, multimodal Digital Twin system built specifically to transform Type 2 Diabetes care from reactive crisis management into proactive, individualized prevention.
> 
> In India today, over 101 million adults live with Type 2 Diabetes, making our country the recognized diabetes capital of the world. But the crisis in India is biochemically distinct. South Asians exhibit what clinicians term the 'Asian Indian Phenotype' — marked by higher visceral adiposity, severe insulin resistance at lower BMIs, and pronounced lipid triad abnormalities. Compounding this, traditional Indian diets heavily rely on refined carbohydrates like white rice and wheat flour rotis, producing sharp postprandial glycemic excursions.
> 
> The central tragedy of current diabetes care is that it is fundamentally **reactive**. Physicians rely on HbA1c tests every three months. But HbA1c is merely a retrospective average. It completely obscures daily glycemic volatility, nocturnal hypoglycemia, and postprandial surges. Interventions only occur after a patient enters the emergency room with blood glucose exceeding 250 or 300 mg/dL, by which point irreversible microvascular damage to the retina, kidneys, and coronary vessels has already taken root.
> 
> The solution is a **Digital Twin**: a dynamic in-silico computational model of the patient that unifies their static medical history with real-time wearable sensor streams to predict adverse glycemic events **two hours before they happen** and simulate corrective actions before the patient suffers a spike."

---

### Part 2: Dual Data Stream Fusion & Feature Pipeline (3:00 - 7:00)
**[Show Slide 3: Multimodal Data Fusion Architecture & Architecture Diagram]**

**Speaker 2:**
> "To build a truly individualized digital twin, the system must bridge two fundamentally disparate data streams:
> 
> 1. **The Static/Historical Stream**: Using clinical protocols inspired by Synthea and real-world Indian diabetic cohorts, we synthesized detailed electronic health records for 200 Indian Type 2 Diabetes patients across metropolitan centers like Mumbai, Delhi, Bangalore, and Pune. This stream captures 46 rigorously preprocessed clinical features — including baseline HbA1c, fasting glucose, ICMR-consensus Asian BMI tiers, duration of disease, KDIGO eGFR renal stages, cardiovascular lipid ratios, and multi-label comorbidity profiles including hypertension, dyslipidemia, and nephropathy.
> 
> 2. **The Dynamic Real-Time Stream**: Interfacing with continuous glucose monitors and consumer IoT wearables, we generated over 1.72 million continuous telemetry readings sampled at 5-minute intervals across 30 days. This includes interstitial glucose, autonomic heart rate, SDNN heart rate variability, 5-minute pedometer counts, and multi-stage sleep architecture. Crucially, we embedded physiological circadian kinetics — including postprandial breakfast, lunch, and dinner surges, dawn phenomenon elevation between 4:00 AM and 8:00 AM, and non-insulin-dependent muscle glucose disposal during activity.
> 
> **Multimodal Data Fusion:**
> To prepare this data for AI modeling without temporal or patient leakage, our `DataFusion` pipeline constructs a 24-step lookback sliding window — representing the patient's preceding 2 hours of telemetry — and broadcast-merges the patient's static clinical embedding. 
> 
> Furthermore, we enforced **strict patient-level cohort partitioning**: 140 patients for training (201,180 sequences), 20 patients for validation (28,740 sequences), and 40 patients for test evaluation (57,480 sequences). Scalers in our `FeatureStore` were fit strictly on training patients, guaranteeing zero data leakage."

---

### Part 3: Hybrid AI Ensemble & Empirical Benchmarks (7:00 - 11:00)
**[Show Slide 4: Hybrid AI Ensemble & Slide 5: Empirical Benchmarks]**

**Speaker 1:**
> "Forecasting blood glucose 2 hours ahead requires capturing both fast dynamic momentum and slow, non-linear clinical interactions. A single model architecture is insufficient. Therefore, we engineered the **GlucoTwin Calibrated Ensemble**, combining two complementary model paradigms:
> 
> **Branch A: PyTorch Multimodal Attention-LSTM (Weight: 0.6)**  
> Our neural branch ingests the temporal sequence tensor of shape (Batch, 24, 54) into a 2-layer Bidirectional LSTM with 128 hidden units. We implemented a temporal self-attention mechanism across all 24 past timesteps. This not only weights the most salient recent glycemic acceleration but also outputs an attention distribution that clinicians can inspect for explainability. The attended sequence vector is fused with an MLP projection of the 46 static EHR biomarkers.
> 
> **Branch B: Tabular XGBoost Classifier (Weight: 0.4)**  
> Our gradient boosting branch extracts 478 engineered tabular features — capturing multi-scale rolling means, standard deviations, acceleration, Glucose Volatility Index, and clinical interaction terms. Trained using the histogram method with cost-sensitive class balancing, it provides razor-sharp boundary discrimination on patient-specific medication regimens and metabolic phenotypes.
> 
> **Empirical Results on 57,480 Held-Out Test Sequences:**
> When evaluated on 40 completely unseen Indian diabetic patients:
> - **ROC-AUC:** GlucoTwin achieved **0.9928** (LSTM: 0.9899, XGBoost: 0.9938).
> - **PR-AUC (Average Precision):** **0.9966**, confirming exceptional precision even under class imbalance.
> - **Overall Accuracy:** **95.34%**.
> - **Sensitivity / Recall:** **95.45%**. In a clinical setting, recall is paramount: less than 4.6% of spikes are missed.
> - **Specificity:** **95.10%**, which minimizes false-alarm fatigue so physicians and patients trust the alerts.
> - **Brier Calibration Score:** **0.0330**, proving that the model's predicted probabilities correspond directly to empirical risk."

---

### Part 4: Digital Twin Engine & 'What-If' Simulation (11:00 - 14:00)
**[Show Slide 6: Digital Twin Engine & Architecture Diagram]**

**Speaker 2:**
> "Predictions alone do not make a digital twin. What elevates GlucoTwin from a predictive model into an authentic in-silico twin is its **physiological state engine and counterfactual simulator**.
> 
> 1. **Multi-Organ Strain Modeling:** The twin continuously evaluates physiological load across three axes:
>    - Pancreatic Beta-cell stress (measuring demand vs endogenous reserve)
>    - Hepatic Glucose Production (HGP velocity during fasting and dawn phases)
>    - Autonomic Cardiovascular strain (derived from nocturnal SDNN HRV and resting heart rate).
> 
> 2. **ADA/RSSDI Clinical Risk Stratification:** Integrating international consensus guidelines, the twin categorizes current risk into Low, Moderate, High, and Critical tiers, dispatching automated alerts for acute hypoglycemia, rapid glycemic velocity, and upcoming spikes.
> 
> 3. **In-Silico 'What-If' Simulation:**
>    Most excitingly, GlucoTwin allows both clinicians and patients to simulate interventions before executing them:
>    - **GLUT4 Physical Walk:** Simulating a 20-minute post-meal brisk walk models non-insulin-mediated GLUT4 translocation, blunting the impending spike by **-48.0 mg/dL** and maintaining euglycemia.
>    - **Pharmacotherapy Timing:** Titrating medication or adjusting dosing timing reduces the excursion by **-38.0 mg/dL**.
>    - **Dietary Fiber Pre-load:** Demonstrates how adding fiber slows gastric emptying, lowering peak elevation by **-18.0 mg/dL**."

---

### Part 5: Live Prototype Demonstration (14:00 - 18:00)
**[Screen Share: Next.js 14 Dashboard running on `http://localhost:3000` & FastAPI on `http://localhost:8000/docs`]**

**Speaker 1:**
> "Now let us demonstrate the live working prototype. On the right, our FastAPI high-performance microservice is running, hosting our trained PyTorch and XGBoost models and servicing requests across all 200 enrolled patients in under 20 milliseconds.
> 
> Here on `http://localhost:3000` is our doctor-facing **Clinical Twin Dashboard**, built with Next.js 14 and Tailwind CSS in a high-contrast dark medical aesthetic.
> 
> Let us select patient **IND-T2D-001 (Rajesh Sharma, 52, Mumbai)** from the cohort dropdown:
> - **PatientOverview Card:** We immediately see his clinical summary — HbA1c of 8.7%, BMI 28.4, established diabetes of 6.5 years on Metformin + Sitagliptin, with comorbid Hypertension and Dyslipidemia.
> - **Glucose Trend Chart:** Powered by Recharts, this interactive display shows 24 hours of 5-minute readings. The green zone represents the target euglycemic range (70-180 mg/dL). Notice the dashed cyan line extending into the future — that is our AI ensemble forecasting a spike up to 215 mg/dL within the next 45 minutes!
> - **RiskGauge:** The animated dial indicates an 88% spike probability with High Risk status and warns of an impending spike in ~35 minutes.
> - **AlertPanel:** Highlights an active Spike Warning with actionable clinical instructions to initiate physical activity.
> - **Twin State & What-If Simulation:** Now watch what happens when we click 'What-If Simulation'. A clinical trial modal appears. When we select 'Go for a 20-minute brisk walk', the twin calculates the counterfactual trajectory in real time: the unmitigated red dashed curve is pulled down into the safe green zone, averting the spike with a peak reduction of 48 mg/dL!"

---

### Part 6: Regulatory Compliance, Clinical Impact & Roadmap (18:00 - 20:00)
**[Show Slide 7: Next.js Dashboard & Slide 8: Impact & Roadmap]**

**Speaker 2:**
> "To address data privacy concerns in accordance with India's **Digital Personal Data Protection (DPDP) Act 2023** and international **HIPAA** standards, GlucoTwin operates entirely within a secure synthetic sandbox. No identifiable personal health information is ever stored or exposed.
> 
> **Clinical & Economic Impact:**
> Chronic diabetes complications account for billions of rupees in out-of-pocket healthcare expenditure across India. By granting patients and endocrinologists a 2-hour predictive window and an in-silico sandbox to test interventions, GlucoTwin directly reduces glycemic variability, blunts postprandial surges, and helps prevent catastrophic microvascular and macrovascular complications.
> 
> **Future Horizon:**
> In Phase 2, we plan to interface GlucoTwin directly via Bluetooth Low Energy with real-world continuous glucose monitors like FreeStyle Libre and Dexcom G7, and in Phase 3, connect our forecasting engine to smart insulin pens and pumps for closed-loop autonomous micro-dosing.
> 
> We thank the Happiest Health jury and look forward to answering your questions!"

---

## 📹 Video Recording Tips for Team Aethan
1. **Screen Recording:** Use OBS Studio, Loom, or Zoom with dual screen sharing (slides + browser).
2. **Audio:** Use a clean external USB microphone or headset in a quiet room.
3. **Resolution:** Record in 1080p (1920x1080) at 30 or 60 fps.
4. **YouTube Upload:** Upload as **Unlisted** on YouTube so only those with the link can view. Add the YouTube URL to the root `README.md`.
