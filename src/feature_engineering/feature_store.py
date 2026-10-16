"""GlucoTwin: Central Feature Store & Pipeline Registry"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.feature_engineering.static_features import (
    extract_static_features,
    generate_synthetic_ehr_cohort,
)
from src.feature_engineering.temporal_features import extract_temporal_features

logger = logging.getLogger("glucotwin.feature_store")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class FeatureStore:
    """Centralized Feature Store for GlucoTwin ML modeling."""

    def __init__(self, store_dir: Optional[Union[str, Path]] = None):
        self.store_dir = Path(store_dir) if store_dir else DATA_PROCESSED_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self.temporal_scaler = StandardScaler()
        self.static_scaler = StandardScaler()
        self.is_fitted: bool = False

        self.dynamic_feature_cols: List[str] = []
        self.static_feature_cols: List[str] = []
        self.target_col: str = "spike_label"

        self.feature_metadata: Dict[str, Any] = {}
        self._register_default_schema()

    def _register_default_schema(self) -> None:
        """Defines default feature categories and schemas."""
        self.dynamic_feature_cols = [
            "glucose",
            "glucose_diff_5m",
            "glucose_rate_of_change",
            "glucose_acceleration",
            "glucose_mean_15m",
            "glucose_std_15m",
            "glucose_diff_15m",
            "glucose_mean_30m",
            "glucose_std_30m",
            "glucose_diff_30m",
            "glucose_mean_1h",
            "glucose_std_1h",
            "glucose_diff_1h",
            "glucose_range_1h",
            "glucose_rolling_mean",
            "glucose_rolling_std",
            "glucose_diff_2h",
            "glucose_range_2h",
            "coefficient_of_variation",
            "time_in_range",
            "time_above_range",
            "glucose_momentum_fast",
            "glucose_momentum_slow",
            "glucose_momentum_divergence",
            "glucose_mag_1h",
            "glucose_volatility_index",
            "postprandial_excursion_magnitude",
            "postprandial_excursion_ratio",
            "sin_time_of_day",
            "cos_time_of_day",
            "is_dawn_period",
            "is_nighttime",
            "sin_day_of_week",
            "cos_day_of_week",
            "is_weekend",
            "estimated_time_since_meal_min",
            "meal_absorption_decay",
            "is_postprandial_window",
            "nocturnal_glucose_mean",
            "dawn_phenomenon_flag",
            "heart_rate",
            "hr_rolling_mean",
            "hr_variability",
            "hr_resting",
            "hr_recovery_delta",
            "steps",
            "steps_hourly",
            "active_minutes",
            "sedentary_minutes",
            "step_count_trend",
            "sleep_duration",
            "sleep_efficiency",
            "deep_sleep_percentage",
            "rem_percentage",
        ]

        self.static_feature_cols = [
            "age",
            "is_male",
            "age_group_code",
            "bmi",
            "bmi_category_code",
            "hba1c",
            "hba1c_tier_code",
            "diabetes_duration_years",
            "duration_risk_code",
            "comorbidity_count",
            "comorbidity_severity_score",
            "has_hypertension",
            "has_dyslipidemia",
            "has_neuropathy",
            "has_retinopathy",
            "has_nephropathy",
            "has_cad",
            "has_nafld",
            "medication_count",
            "medication_complexity_score",
            "uses_insulin",
            "uses_metformin",
            "uses_sulfonylurea",
            "uses_sglt2",
            "uses_dpp4",
            "uses_statin",
            "uses_antihypertensive",
            "lifestyle_composite_score",
            "daily_protein_intake_g",
            "sleep_hours_avg",
            "stress_level",
            "egfr",
            "egfr_stage_code",
            "blood_pressure_systolic",
            "blood_pressure_diastolic",
            "cholesterol_total",
            "cholesterol_ldl",
            "cholesterol_hdl",
            "triglycerides",
            "tg_to_hdl_ratio",
            "cardiovascular_risk_composite",
            "is_vegetarian",
            "protein_intake_adequacy",
            "carb_protein_imbalance",
            "south_asian_phenotype_score",
            "urban_stress_index",
        ]

        self.feature_metadata = {
            "dynamic_features_count": len(self.dynamic_feature_cols),
            "static_features_count": len(self.static_feature_cols),
            "target": self.target_col,
            "sampling_interval": "5min",
            "lookback_sequence_length": 24,
            "prediction_horizon_steps": 24,
            "prediction_horizon_minutes": 120,
            "spike_threshold_mg_dl": 180.0,
            "cohort_origin": "Indian Type 2 Diabetes Cohort (Amity University Mumbai)",
        }

    def build_features(
        self, ehr_df: pd.DataFrame, cgm_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Executes static and temporal feature extraction on raw or processed inputs."""
        logger.info("Building static and temporal feature sets...")
        static_feats = extract_static_features(ehr_df)
        temporal_feats = extract_temporal_features(cgm_df)

        # Retain only recognized numeric columns that actually exist
        valid_dynamic = [c for c in self.dynamic_feature_cols if c in temporal_feats.columns]
        valid_static = [c for c in self.static_feature_cols if c in static_feats.columns]

        self.dynamic_feature_cols = valid_dynamic
        self.static_feature_cols = valid_static
        self.feature_metadata["dynamic_features_count"] = len(self.dynamic_feature_cols)
        self.feature_metadata["static_features_count"] = len(self.static_feature_cols)

        logger.info(
            f"Feature extraction complete. Dynamic features: {len(valid_dynamic)}, "
            f"Static features: {len(valid_static)}"
        )
        return static_feats, temporal_feats

    def fit(
        self,
        static_df: pd.DataFrame,
        temporal_df: pd.DataFrame,
        train_patient_ids: List[str],
    ) -> "FeatureStore":
        """Fits StandardScalers strictly on the subset of data belonging to train_patient_ids."""
        logger.info(
            f"Fitting FeatureStore scalers on {len(train_patient_ids)} training patients..."
        )
        train_pids_set = set(str(p) for p in train_patient_ids)

        # Fit static scaler
        train_static = static_df[static_df["patient_id"].astype(str).isin(train_pids_set)]
        if len(train_static) == 0:
            raise ValueError("No matching training patients found in static DataFrame.")

        static_mat = train_static[self.static_feature_cols].fillna(train_static[self.static_feature_cols].median()).values
        self.static_scaler.fit(static_mat)

        # Fit temporal scaler
        train_temporal = temporal_df[temporal_df["patient_id"].astype(str).isin(train_pids_set)]
        if len(train_temporal) == 0:
            raise ValueError("No matching training patients found in temporal DataFrame.")

        temp_mat = train_temporal[self.dynamic_feature_cols].fillna(train_temporal[self.dynamic_feature_cols].median()).values
        self.temporal_scaler.fit(temp_mat)

        self.is_fitted = True
        logger.info("FeatureStore successfully fitted.")
        return self

    def transform_static(self, static_df: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Transforms static features using the fitted scaler."""
        if scale and not self.is_fitted:
            raise RuntimeError("FeatureStore must be fitted before calling transform_static with scale=True.")

        mat = static_df[self.static_feature_cols].fillna(0.0).values
        if scale:
            mat = self.static_scaler.transform(mat)
        return mat

    def transform_temporal(self, temporal_df: pd.DataFrame, scale: bool = True) -> np.ndarray:
        """Transforms temporal features using the fitted scaler."""
        if scale and not self.is_fitted:
            raise RuntimeError("FeatureStore must be fitted before calling transform_temporal with scale=True.")

        mat = temporal_df[self.dynamic_feature_cols].fillna(0.0).values
        if scale:
            mat = self.temporal_scaler.transform(mat)
        return mat

    def fit_transform(
        self,
        static_df: pd.DataFrame,
        temporal_df: pd.DataFrame,
        train_patient_ids: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fits on training patients and transforms the provided static and temporal datasets."""
        self.fit(static_df, temporal_df, train_patient_ids)
        scaled_static = self.transform_static(static_df, scale=True)
        scaled_temporal = self.transform_temporal(temporal_df, scale=True)
        return scaled_static, scaled_temporal

    def get_feature_names(self) -> Dict[str, List[str]]:
        """Returns ordered lists of dynamic and static feature names."""
        return {
            "dynamic_features": list(self.dynamic_feature_cols),
            "static_features": list(self.static_feature_cols),
            "target": [self.target_col],
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns feature metadata and configuration parameters."""
        meta = dict(self.feature_metadata)
        meta["is_fitted"] = self.is_fitted
        meta["dynamic_features"] = self.dynamic_feature_cols
        meta["static_features"] = self.static_feature_cols
        return meta

    def save(self, filepath: Optional[Union[str, Path]] = None) -> Path:
        """Saves fitted scalers, feature column lists, and metadata to disk."""
        save_file = Path(filepath) if filepath else (self.store_dir / "feature_store.joblib")
        save_file.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "temporal_scaler": self.temporal_scaler,
            "static_scaler": self.static_scaler,
            "is_fitted": self.is_fitted,
            "dynamic_feature_cols": self.dynamic_feature_cols,
            "static_feature_cols": self.static_feature_cols,
            "target_col": self.target_col,
            "feature_metadata": self.feature_metadata,
        }

        joblib.dump(payload, save_file)
        logger.info(f"Saved FeatureStore artifact to {save_file}.")

        # Also write a human-readable JSON schema
        json_file = save_file.with_suffix(".json")
        with open(json_file, "w") as f:
            json.dump(self.get_metadata(), f, indent=2)

        return save_file

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "FeatureStore":
        """Loads a persisted FeatureStore instance from disk."""
        load_file = Path(filepath)
        if not load_file.exists():
            raise FileNotFoundError(f"FeatureStore artifact not found at {load_file}")

        payload = joblib.load(load_file)
        store = cls(store_dir=load_file.parent)
        store.temporal_scaler = payload["temporal_scaler"]
        store.static_scaler = payload["static_scaler"]
        store.is_fitted = payload["is_fitted"]
        store.dynamic_feature_cols = payload["dynamic_feature_cols"]
        store.static_feature_cols = payload["static_feature_cols"]
        store.target_col = payload.get("target_col", "spike_label")
        store.feature_metadata = payload.get("feature_metadata", {})

        logger.info(f"Loaded FeatureStore artifact from {load_file}.")
        return store


if __name__ == "__main__":
    print("Testing FeatureStore module...")
    store = FeatureStore()
    ehr = generate_synthetic_ehr_cohort(num_patients=10)
    dates = pd.date_range("2026-03-01", periods=288, freq="5min")
    cgm_list = []
    for pid in ehr["patient_id"]:
        cgm_list.append(pd.DataFrame({
            "patient_id": [pid] * 288,
            "timestamp": dates,
            "glucose": 140.0 + 30.0 * np.sin(np.linspace(0, 4 * np.pi, 288)) + np.random.normal(0, 5, 288),
            "heart_rate": 70.0 + 10.0 * np.random.rand(288),
            "steps": np.random.poisson(25, 288),
        }))
    cgm = pd.concat(cgm_list, axis=0).reset_index(drop=True)

    static_df, temp_df = store.build_features(ehr, cgm)
    train_ids = list(ehr["patient_id"][:7])
    store.fit(static_df, temp_df, train_ids)
    saved_path = store.save()
    print("Saved FeatureStore to:", saved_path)
    loaded_store = FeatureStore.load(saved_path)
    print("Loaded FeatureStore successfully. Is fitted:", loaded_store.is_fitted)
    print("Dynamic features count:", len(loaded_store.dynamic_feature_cols))
    print("Static features count:", len(loaded_store.static_feature_cols))
    print("FeatureStore self-test passed!")
