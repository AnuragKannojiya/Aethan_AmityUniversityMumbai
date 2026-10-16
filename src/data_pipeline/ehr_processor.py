"""
GlucoTwin EHR Data Processor
============================

Comprehensive preprocessing and feature engineering pipeline for Electronic
Health Records (EHR) of Type 2 Diabetes patients.

Responsibilities:
    1. Ingestion: Load raw patient EHR data (ehr_patients.csv)
    2. Data Hygiene: Validate clinical ranges, impute missing values, enforce types
    3. Categorical Encoding:
        - OneHotEncoder for medication regimens
        - LabelEncoder for ordinal and nominal clinical/demographic features
        - Multi-label binary encoding for comorbidities
    4. Numerical Standardization:
        - StandardScaler normalization for continuous physiological biomarkers
    5. Clinical Risk Stratification:
        - Algorithmic scoring into Low, Medium, High, and Critical risk tiers
          based on ADA/RSSDI glycemic and comorbidity guidelines
    6. Export & Serialization:
        - Processed tabular CSV: data/processed/ehr_processed.csv
        - Feature array: data/processed/ehr_features.npy
        - Model artifacts: Scalers, encoders, and feature metadata via joblib

Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("EHRProcessor")

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ehr_patients.csv"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ==============================================================================
# CLINICAL SCHEMA CONSTANTS
# ==============================================================================

NUMERICAL_FEATURES = [
    "age",
    "bmi",
    "weight_kg",
    "height_cm",
    "hba1c",
    "fasting_glucose",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "cholesterol_total",
    "cholesterol_ldl",
    "cholesterol_hdl",
    "triglycerides",
    "creatinine",
    "egfr",
    "diabetes_duration_years",
    "sleep_hours_avg",
    "genetic_risk_score"
]

CATEGORICAL_FEATURES = [
    "gender",
    "city",
    "smoking_status",
    "physical_activity_level",
    "diet_type",
    "stress_level",
    "family_history_diabetes",
    "tcf7l2_rs7903146",
    "pparg_rs1801282",
    "kcnj11_rs5219",
    "slc30a8_rs13266634",
    "fto_rs9939609",
    "hla_dq_risk"
]

KNOWN_COMORBIDITIES = [
    "Hypertension",
    "Dyslipidemia",
    "Neuropathy",
    "Retinopathy",
    "CKD"
]

RISK_TIERS = ["Low", "Medium", "High", "Critical"]
RISK_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

# Clinical bounds for plausibility checks and outlier clipping
CLINICAL_BOUNDS = {
    "age": (18.0, 100.0),
    "bmi": (15.0, 60.0),
    "weight_kg": (30.0, 220.0),
    "height_cm": (120.0, 220.0),
    "hba1c": (4.0, 18.0),
    "fasting_glucose": (50.0, 500.0),
    "blood_pressure_systolic": (70.0, 240.0),
    "blood_pressure_diastolic": (40.0, 150.0),
    "cholesterol_total": (80.0, 500.0),
    "cholesterol_ldl": (30.0, 350.0),
    "cholesterol_hdl": (15.0, 120.0),
    "triglycerides": (40.0, 800.0),
    "creatinine": (0.3, 6.0),
    "egfr": (10.0, 150.0),
    "diabetes_duration_years": (0.0, 60.0),
    "sleep_hours_avg": (2.0, 14.0),
    "genetic_risk_score": (0.0, 1.0)
}


# ==============================================================================
# EHR PROCESSOR CLASS
# ==============================================================================

class EHRProcessor:
    """
    Electronic Health Record (EHR) processing and feature engineering engine.
    Handles data validation, categorical encoding, standardization, risk
    stratification, and serialization for downstream ML models and inference.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        scaler_type: str = "standard"
    ):
        """
        Initialize the EHRProcessor.

        Args:
            output_dir: Directory where processed files and transformer artifacts
                        will be saved. Defaults to `data/processed`.
            scaler_type: Type of numerical scaler to use ('standard' for StandardScaler).
        """
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scaler_type = scaler_type
        self.scaler = StandardScaler()
        
        # Categorical encoders
        self.label_encoders: Dict[str, LabelEncoder] = {
            col: LabelEncoder() for col in CATEGORICAL_FEATURES
        }
        self.medication_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        self.is_fitted: bool = False
        self.feature_names: List[str] = []
        self.medication_feature_names: List[str] = []
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.feature_matrix: Optional[np.ndarray] = None

    # --------------------------------------------------------------------------
    # 1. LOAD
    # --------------------------------------------------------------------------
    def load(self, filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Load raw EHR dataset from CSV.

        Args:
            filepath: Path to ehr_patients.csv. Defaults to data/raw/ehr_patients.csv.

        Returns:
            pd.DataFrame: Loaded raw EHR data.
        """
        path = Path(filepath) if filepath else DEFAULT_RAW_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Raw EHR file not found at: {path}. "
                f"Please run 'python data/generate_synthetic.py' first."
            )

        logger.info(f"Loading raw EHR dataset from: {path}")
        df = pd.read_csv(path)
        self.raw_df = df.copy()
        logger.info(f"Successfully loaded {len(df)} patient records with {df.shape[1]} columns.")
        return df

    # --------------------------------------------------------------------------
    # 2. CLEAN & VALIDATE
    # --------------------------------------------------------------------------
    def clean(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean and validate EHR records.
        - Fills missing values with clinically sound defaults
        - Enforces valid clinical ranges by clipping extreme non-physiological outliers
        - Normalizes boolean and categorical strings
        - Ensures standard datetime format for longitudinal records

        Args:
            df: DataFrame to clean. If None, uses loaded raw_df.

        Returns:
            pd.DataFrame: Cleaned patient DataFrame.
        """
        if df is None:
            if self.raw_df is None:
                self.load()
            df = self.raw_df.copy()
        else:
            df = df.copy()

        logger.info("Cleaning and validating EHR dataset...")

        # Handle null comorbidities
        if "comorbidities" in df.columns:
            df["comorbidities"] = df["comorbidities"].fillna("None").astype(str).str.strip()

        # Default genetic markers if omitted in single-record inference payloads
        genetic_defaults = {
            "tcf7l2_rs7903146": "CT",
            "pparg_rs1801282": "CC",
            "kcnj11_rs5219": "CT",
            "slc30a8_rs13266634": "CT",
            "fto_rs9939609": "TA",
            "hla_dq_risk": False,
            "genetic_risk_score": 0.42,
        }
        for g_col, g_def in genetic_defaults.items():
            if g_col not in df.columns:
                df[g_col] = g_def

        # Enforce clinical bounds and impute missing numericals
        for col in NUMERICAL_FEATURES:
            if col in df.columns:
                # Impute missing with median
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    logger.warning(f"Imputed {df[col].isnull().sum()} missing values in column '{col}' with median {median_val}.")
                
                # Clip to physiological bounds
                if col in CLINICAL_BOUNDS:
                    low_b, high_b = CLINICAL_BOUNDS[col]
                    df[col] = df[col].clip(lower=low_b, upper=high_b)
            else:
                logger.warning(f"Expected numerical column '{col}' missing from EHR data.")

        # Ensure boolean family history is clean
        if "family_history_diabetes" in df.columns:
            df["family_history_diabetes"] = df["family_history_diabetes"].map(
                lambda x: True if str(x).lower() in ["true", "1", "yes", "t"] else False
            )

        # Standardize categorical strings
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col in df.columns and cat_col != "family_history_diabetes":
                df[cat_col] = df[cat_col].fillna("Unknown").astype(str).str.strip()

        # Validate dates
        for date_col in ["diagnosis_date", "last_visit_date"]:
            if date_col in df.columns:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

        self.cleaned_df = df.copy()
        logger.info(f"Cleaning complete. Validated {len(df)} patient records.")
        return df

    # --------------------------------------------------------------------------
    # 3. CLINICAL RISK STRATIFICATION
    # --------------------------------------------------------------------------
    def compute_risk_stratification(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Stratify patients into clinical risk tiers: Low, Medium, High, Critical.
        Based on Indian (RSSDI) & International (ADA) Type 2 Diabetes guidelines.

        Scoring Logic:
            1. HbA1c:
               - < 7.0%: 0 pts (Controlled)
               - 7.0 - 8.0%: 1 pt (Moderate)
               - 8.0 - 9.0%: 2 pts (Sub-optimal)
               - 9.0 - 10.0%: 3 pts (High risk)
               - >= 10.0%: 4 pts (Severe / Critical)
            2. Fasting Glucose:
               - < 130 mg/dL: 0 pts
               - 130 - 175 mg/dL: 1 pt
               - 175 - 230 mg/dL: 2 pts
               - >= 230 mg/dL: 3 pts
            3. BMI:
               - < 25.0: 0 pts
               - 25.0 - 30.0: 1 pt
               - >= 30.0: 2 pts
            4. Comorbidities:
               - Hypertension / Dyslipidemia: 1 pt each
               - Organ damage (Neuropathy / Retinopathy / CKD): 2 pts each
            5. Duration:
               - >= 10 years: 1 pt

        Risk Tier Mapping:
            - 0 - 2 pts: Low
            - 3 - 5 pts: Medium
            - 6 - 8 pts: High
            - >= 9 pts: Critical

        Args:
            df: Cleaned patient DataFrame.

        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]:
                - risk_category: Categorical text ('Low', 'Medium', 'High', 'Critical')
                - risk_score: Numeric integer score
                - risk_class: Integer class index (0, 1, 2, 3)
        """
        scores = np.zeros(len(df), dtype=int)

        # 1. HbA1c points
        hba1c = df["hba1c"].values
        scores += np.where(hba1c >= 10.0, 4,
                  np.where(hba1c >= 9.0, 3,
                  np.where(hba1c >= 8.0, 2,
                  np.where(hba1c >= 7.0, 1, 0))))

        # 2. Fasting Glucose points
        fg = df["fasting_glucose"].values
        scores += np.where(fg >= 230.0, 3,
                  np.where(fg >= 175.0, 2,
                  np.where(fg >= 130.0, 1, 0)))

        # 3. BMI points (Asian Indian thresholds)
        bmi = df["bmi"].values
        scores += np.where(bmi >= 30.0, 2,
                  np.where(bmi >= 25.0, 1, 0))

        # 4. Comorbidities points
        comorb_series = df["comorbidities"].fillna("None").astype(str)
        for _, comorb_str in comorb_series.items():
            pass  # Vectorized below
        
        has_htn = comorb_series.str.contains("Hypertension", case=False).astype(int).values
        has_dys = comorb_series.str.contains("Dyslipidemia", case=False).astype(int).values
        has_neuro = comorb_series.str.contains("Neuropathy", case=False).astype(int).values
        has_retino = comorb_series.str.contains("Retinopathy", case=False).astype(int).values
        has_ckd = comorb_series.str.contains("CKD", case=False).astype(int).values

        scores += (has_htn * 1) + (has_dys * 1) + (has_neuro * 2) + (has_retino * 2) + (has_ckd * 2)

        # 5. Long duration points
        dur = df["diabetes_duration_years"].values
        scores += np.where(dur >= 10.0, 1, 0)

        # Stratification assignment
        risk_category = np.where(scores >= 9, "Critical",
                        np.where(scores >= 6, "High",
                        np.where(scores >= 3, "Medium", "Low")))
        
        risk_class = np.array([RISK_MAP[cat] for cat in risk_category], dtype=int)

        return (
            pd.Series(risk_category, index=df.index, name="risk_stratification"),
            pd.Series(scores, index=df.index, name="risk_score"),
            pd.Series(risk_class, index=df.index, name="risk_class")
        )

    # --------------------------------------------------------------------------
    # 4. ENCODE CATEGORICALS & COMORBIDITIES
    # --------------------------------------------------------------------------
    def encode(self, df: Optional[pd.DataFrame] = None, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features, medication regimens, and comorbidities.

        - OneHotEncoder on 'medication'
        - LabelEncoder on categorical features
        - Multi-label binary decomposition for 'comorbidities'
        - Appends risk stratification

        Args:
            df: Cleaned patient DataFrame.
            fit: Whether to fit encoders on this dataset.

        Returns:
            pd.DataFrame: DataFrame containing transformed categorical features.
        """
        if df is None:
            if self.cleaned_df is None:
                self.clean()
            df = self.cleaned_df.copy()
        else:
            df = df.copy()

        logger.info(f"Encoding categorical variables (fit={fit})...")

        # 1. Comorbidities multi-label binary encoding
        comorb_series = df["comorbidities"].fillna("None").astype(str)
        comorb_count = np.zeros(len(df), dtype=int)

        for comorb in KNOWN_COMORBIDITIES:
            has_comorb = comorb_series.str.contains(comorb, case=False).astype(int)
            df[f"comorbidity_{comorb}"] = has_comorb
            comorb_count += has_comorb.values
        
        df["comorbidity_count"] = comorb_count

        # 2. Medication OneHotEncoding
        med_input = df[["medication"]].fillna("None").astype(str)
        if fit:
            med_encoded = self.medication_encoder.fit_transform(med_input)
            raw_cats = self.medication_encoder.categories_[0]
            # Standardize feature names: remove spaces and special characters
            self.medication_feature_names = [
                f"med_{c.replace(' + ', '_').replace(' ', '_')}" for c in raw_cats
            ]
        else:
            med_encoded = self.medication_encoder.transform(med_input)

        for idx, col_name in enumerate(self.medication_feature_names):
            df[col_name] = med_encoded[:, idx].astype(float)

        # 3. Categorical LabelEncoding
        for cat_col in CATEGORICAL_FEATURES:
            if cat_col in df.columns:
                series_str = df[cat_col].astype(str)
                encoded_col_name = f"{cat_col}_encoded"
                if fit:
                    self.label_encoders[cat_col].fit(series_str)
                    df[encoded_col_name] = self.label_encoders[cat_col].transform(series_str)
                else:
                    # Safe transform for unseen labels during inference
                    known_classes = set(self.label_encoders[cat_col].classes_)
                    series_safe = series_str.map(
                        lambda s: s if s in known_classes else self.label_encoders[cat_col].classes_[0]
                    )
                    df[encoded_col_name] = self.label_encoders[cat_col].transform(series_safe)

        # 4. Risk Stratification
        risk_cat, risk_score, risk_cls = self.compute_risk_stratification(df)
        df["risk_stratification"] = risk_cat
        df["risk_score"] = risk_score
        df["risk_class"] = risk_cls

        logger.info(
            f"Risk stratification distribution:\n{df['risk_stratification'].value_counts().to_dict()}"
        )

        return df

    # --------------------------------------------------------------------------
    # 5. NORMALIZE NUMERICAL FEATURES
    # --------------------------------------------------------------------------
    def normalize(
        self,
        df: Optional[pd.DataFrame] = None,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Standardize continuous numerical biomarker features using StandardScaler.

        Args:
            df: DataFrame containing cleaned and encoded features.
            fit: Whether to fit the scaler (True for training/batch, False for test/inference).

        Returns:
            Tuple[pd.DataFrame, np.ndarray]:
                - df: DataFrame with added normalized columns (*_norm)
                - feature_matrix: 2D numpy array of all model-ready features (N x D)
        """
        if df is None:
            df = self.encode(fit=fit)
        else:
            df = df.copy()

        logger.info(f"Standardizing continuous numerical features (fit={fit})...")

        num_vals = df[NUMERICAL_FEATURES].values.astype(float)

        if fit:
            scaled_num_vals = self.scaler.fit_transform(num_vals)
            self.is_fitted = True
        else:
            scaled_num_vals = self.scaler.transform(num_vals)

        # Store normalized columns in dataframe
        for idx, col in enumerate(NUMERICAL_FEATURES):
            df[f"{col}_norm"] = scaled_num_vals[:, idx]

        # Assemble full numeric feature matrix for downstream ML models
        # Columns included in the feature matrix:
        #   - Normalized continuous features: NUMERICAL_FEATURES (_norm)
        #   - One-hot encoded medications: medication_feature_names
        #   - Multi-label comorbidities: comorbidity_<Name>, comorbidity_count
        #   - Encoded categoricals: <cat>_encoded
        feature_cols = []

        # 1. Normalized continuous
        for col in NUMERICAL_FEATURES:
            feature_cols.append(f"{col}_norm")

        # 2. Medication one-hot
        feature_cols.extend(self.medication_feature_names)

        # 3. Comorbidities
        for comorb in KNOWN_COMORBIDITIES:
            feature_cols.append(f"comorbidity_{comorb}")
        feature_cols.append("comorbidity_count")

        # 4. Encoded categoricals
        for cat_col in CATEGORICAL_FEATURES:
            feature_cols.append(f"{cat_col}_encoded")

        if fit:
            self.feature_names = feature_cols

        feature_matrix = df[self.feature_names].values.astype(np.float32)

        self.processed_df = df
        self.feature_matrix = feature_matrix

        logger.info(
            f"Normalized feature matrix generated with shape: {feature_matrix.shape} "
            f"({len(self.feature_names)} features)."
        )

        return df, feature_matrix

    # --------------------------------------------------------------------------
    # 6. GET FEATURE MATRIX
    # --------------------------------------------------------------------------
    def get_features(self, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Retrieve pure numerical feature matrix X ready for PyTorch / XGBoost.

        Args:
            df: Optional DataFrame to extract features from. If None, returns
                internally cached feature matrix.

        Returns:
            np.ndarray: Feature matrix of shape (N, D), dtype float32.
        """
        if df is None:
            if self.feature_matrix is None:
                _, matrix = self.normalize(fit=True)
                return matrix
            return self.feature_matrix
        
        # If external df provided, transform it with fitted scalers
        if not self.is_fitted:
            raise RuntimeError("Processor is not fitted yet. Call process_all() or fit first.")
        
        cleaned = self.clean(df)
        encoded = self.encode(cleaned, fit=False)
        _, matrix = self.normalize(encoded, fit=False)
        return matrix

    # --------------------------------------------------------------------------
    # 7. ARTIFACT PERSISTENCE & SERIALIZATION
    # --------------------------------------------------------------------------
    def save_artifacts(
        self,
        processed_df: Optional[pd.DataFrame] = None,
        feature_matrix: Optional[np.ndarray] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, Path]:
        """
        Save processed dataset, feature matrix, and serialization artifacts to disk.

        Saved Files:
            - data/processed/ehr_processed.csv
            - data/processed/ehr_features.npy
            - data/processed/ehr_scaler.joblib
            - data/processed/ehr_encoders.joblib
            - data/processed/ehr_feature_names.joblib
            - data/processed/ehr_metadata.json

        Args:
            processed_df: Processed DataFrame. Defaults to self.processed_df.
            feature_matrix: NumPy feature matrix. Defaults to self.feature_matrix.
            output_dir: Destination directory. Defaults to self.output_dir.

        Returns:
            Dict[str, Path]: Map of artifact names to file paths.
        """
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        df = processed_df if processed_df is not None else self.processed_df
        mat = feature_matrix if feature_matrix is not None else self.feature_matrix

        if df is None or mat is None:
            raise ValueError("No processed data available to save. Run normalize() or process_all() first.")

        csv_path = out_dir / "ehr_processed.csv"
        npy_path = out_dir / "ehr_features.npy"
        scaler_path = out_dir / "ehr_scaler.joblib"
        encoders_path = out_dir / "ehr_encoders.joblib"
        feature_names_path = out_dir / "ehr_feature_names.joblib"
        metadata_path = out_dir / "ehr_metadata.json"

        # 1. Save processed CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved processed EHR dataset ({len(df)} rows) to: {csv_path}")

        # 2. Save feature matrix npy
        np.save(npy_path, mat)
        logger.info(f"Saved feature matrix {mat.shape} to: {npy_path}")

        # 3. Save scaler
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to: {scaler_path}")

        # 4. Save categorical & medication encoders
        encoders_dict = {
            "label_encoders": self.label_encoders,
            "medication_encoder": self.medication_encoder,
            "medication_feature_names": self.medication_feature_names,
            "known_comorbidities": KNOWN_COMORBIDITIES,
            "risk_map": RISK_MAP
        }
        joblib.dump(encoders_dict, encoders_path)
        logger.info(f"Saved encoders to: {encoders_path}")

        # 5. Save feature names
        joblib.dump(self.feature_names, feature_names_path)
        logger.info(f"Saved {len(self.feature_names)} feature names to: {feature_names_path}")

        # 6. Save JSON summary metadata
        risk_dist = df["risk_stratification"].value_counts().to_dict()
        metadata = {
            "n_patients": len(df),
            "n_features": mat.shape[1],
            "feature_names": self.feature_names,
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "risk_distribution": risk_dist,
            "hba1c_mean": round(float(df["hba1c"].mean()), 2),
            "fasting_glucose_mean": round(float(df["fasting_glucose"].mean()), 2),
            "bmi_mean": round(float(df["bmi"].mean()), 2),
            "processed_timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "processed_csv": csv_path,
            "features_npy": npy_path,
            "scaler": scaler_path,
            "encoders": encoders_path,
            "feature_names": feature_names_path,
            "metadata": metadata_path
        }

    # --------------------------------------------------------------------------
    # 8. LOAD SAVED ARTIFACTS (INFERENCE SUPPORT)
    # --------------------------------------------------------------------------
    def load_artifacts(self, artifact_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Load serialized scalers, encoders, and feature names from disk for inference.

        Args:
            artifact_dir: Directory containing saved artifacts. Defaults to self.output_dir.
        """
        art_dir = Path(artifact_dir) if artifact_dir else self.output_dir

        scaler_path = art_dir / "ehr_scaler.joblib"
        encoders_path = art_dir / "ehr_encoders.joblib"
        feature_names_path = art_dir / "ehr_feature_names.joblib"

        if not scaler_path.exists() or not encoders_path.exists():
            raise FileNotFoundError(f"Transformer artifacts not found in: {art_dir}")

        self.scaler = joblib.load(scaler_path)
        encoders_dict = joblib.load(encoders_path)

        self.label_encoders = encoders_dict["label_encoders"]
        self.medication_encoder = encoders_dict["medication_encoder"]
        self.medication_feature_names = encoders_dict["medication_feature_names"]
        self.feature_names = joblib.load(feature_names_path)
        self.is_fitted = True

        logger.info(f"Successfully loaded EHRProcessor artifacts from: {art_dir}")

    # --------------------------------------------------------------------------
    # 9. SINGLE PATIENT INFERENCE TRANSFORM
    # --------------------------------------------------------------------------
    def transform_single(self, patient_dict: Dict[str, Any]) -> np.ndarray:
        """
        Transform a single raw patient dictionary into a model-ready feature vector.
        Essential for the Digital Twin API and real-time spike simulation.

        Args:
            patient_dict: Key-value dictionary containing patient parameters
                          (e.g., {'age': 52, 'hba1c': 8.2, 'medication': 'Metformin', ...}).

        Returns:
            np.ndarray: 2D feature array of shape (1, D), dtype float32.
        """
        if not self.is_fitted:
            # Try to load existing artifacts if present
            self.load_artifacts()

        df_single = pd.DataFrame([patient_dict])
        cleaned = self.clean(df_single)
        encoded = self.encode(cleaned, fit=False)
        _, matrix = self.normalize(encoded, fit=False)
        return matrix

    # --------------------------------------------------------------------------
    # 10. MASTER PIPELINE EXECUTION
    # --------------------------------------------------------------------------
    def process_all(
        self,
        raw_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Execute end-to-end EHR processing pipeline:
            load -> clean -> stratify -> encode -> normalize -> save_artifacts.

        Args:
            raw_path: Path to raw ehr_patients.csv.
            output_dir: Destination path for processed data and artifacts.

        Returns:
            Tuple[pd.DataFrame, np.ndarray]:
                - processed_df: Fully processed DataFrame.
                - feature_matrix: Normalized float32 feature matrix.
        """
        logger.info("=" * 80)
        logger.info("Starting GlucoTwin EHR Processing Pipeline")
        logger.info("=" * 80)

        # 1. Load
        self.load(filepath=raw_path)

        # 2. Clean & Validate
        self.clean()

        # 3 & 4. Encode & Normalize
        df_norm, feat_mat = self.normalize(fit=True)

        # 5. Save Artifacts
        artifacts = self.save_artifacts(
            processed_df=df_norm,
            feature_matrix=feat_mat,
            output_dir=output_dir
        )

        logger.info("=" * 80)
        logger.info("EHR Processing Pipeline Completed Successfully!")
        logger.info(f"  - Patients Processed: {len(df_norm)}")
        logger.info(f"  - Features Generated: {feat_mat.shape[1]}")
        logger.info(f"  - Processed CSV:      {artifacts['processed_csv']}")
        logger.info(f"  - Feature Array:      {artifacts['features_npy']}")
        logger.info(f"  - Scaler / Encoders:  {artifacts['scaler']} / {artifacts['encoders']}")
        logger.info("=" * 80)

        return df_norm, feat_mat


# ==============================================================================
# CLI EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    processor = EHRProcessor()
    processor.process_all()
