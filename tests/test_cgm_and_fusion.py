"""
Unit & Integration Tests for CGM Processor, Feature Engineering & Data Fusion
=============================================================================
Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline.cgm_processor import CGMProcessor
from src.data_pipeline.data_fusion import DataFusion
from src.feature_engineering.feature_store import FeatureStore
from src.feature_engineering.static_features import (
    extract_static_features,
    generate_synthetic_ehr_cohort,
)
from src.feature_engineering.temporal_features import extract_temporal_features


class TestCGMProcessor(unittest.TestCase):
    """Test suite for CGMProcessor time-series pipeline."""

    @classmethod
    def setUpClass(cls):
        # Generate small synthetic CGM dataset for 3 patients over 2 days
        cls.raw_df = CGMProcessor.generate_synthetic_cgm(num_patients=3, days=2, seed=42)
        cls.processor = CGMProcessor()
        cls.processor.load(cls.raw_df)

    def test_load_and_columns(self):
        """Verifies loading, column mapping, and patient IDs."""
        self.assertIsNotNone(self.processor.df)
        self.assertEqual(self.processor.df["patient_id"].nunique(), 3)
        self.assertIn("glucose", self.processor.df.columns)
        self.assertIn("heart_rate", self.processor.df.columns)
        self.assertIn("steps", self.processor.df.columns)

    def test_clean_bounds(self):
        """Verifies physiological cleaning and artifact removal."""
        cleaned = self.processor.clean()
        self.assertTrue((cleaned["glucose"] >= 30.0).all())
        self.assertTrue((cleaned["glucose"] <= 500.0).all())
        self.assertTrue((cleaned["heart_rate"] >= 35.0).all())
        self.assertTrue((cleaned["heart_rate"] <= 210.0).all())
        self.assertTrue((cleaned["steps"] >= 0.0).all())
        self.assertFalse(cleaned["glucose"].isna().any())

    def test_resample(self):
        """Verifies consistent 5-minute sampling frequency."""
        resampled = self.processor.resample(interval="5min")
        for pid in resampled["patient_id"].unique():
            sub_ts = resampled[resampled["patient_id"] == pid]["timestamp"]
            diffs = sub_ts.diff().dropna()
            self.assertTrue((diffs == pd.Timedelta(minutes=5)).all())

    def test_glucose_metrics(self):
        """Verifies 2h rolling glucose metrics."""
        df_g = self.processor.compute_glucose_metrics(window="2h")
        self.assertIn("glucose_rolling_mean", df_g.columns)
        self.assertIn("glucose_rolling_std", df_g.columns)
        self.assertIn("glucose_rate_of_change", df_g.columns)
        self.assertIn("glucose_acceleration", df_g.columns)
        self.assertIn("time_in_range", df_g.columns)
        self.assertIn("coefficient_of_variation", df_g.columns)
        self.assertIn("glucose_trend", df_g.columns)
        self.assertTrue((df_g["time_in_range"] >= 0.0).all() and (df_g["time_in_range"] <= 100.0).all())

    def test_hr_and_activity_metrics(self):
        """Verifies 1h rolling HR and activity features."""
        df_hr = self.processor.compute_hr_metrics(window="1h")
        df_act = self.processor.compute_activity_metrics(window="1h")
        self.assertIn("hr_rolling_mean", df_hr.columns)
        self.assertIn("hr_variability", df_hr.columns)
        self.assertIn("hr_resting", df_hr.columns)
        self.assertIn("steps_hourly", df_act.columns)
        self.assertIn("active_minutes", df_act.columns)
        self.assertIn("sedentary_minutes", df_act.columns)

    def test_spike_labeling(self):
        """Verifies binary spike labeling and forward-looking horizon."""
        df_spikes = self.processor.create_spike_labels(threshold=180.0, lookahead="2h")
        self.assertIn("spike_label", df_spikes.columns)
        self.assertIn("future_peak_glucose", df_spikes.columns)
        # Check binary labels 0 or 1
        unique_labels = set(df_spikes["spike_label"].unique())
        self.assertTrue(unique_labels.issubset({0, 1}))


class TestFeatureEngineering(unittest.TestCase):
    """Test suite for static and temporal feature extractors."""

    def test_static_feature_extraction(self):
        """Verifies Indian T2D clinical static feature extraction."""
        ehr_cohort = generate_synthetic_ehr_cohort(num_patients=5, seed=101)
        feats = extract_static_features(ehr_cohort)

        self.assertEqual(len(feats), 5)
        # Check age group buckets
        self.assertIn("age_group_code", feats.columns)
        self.assertIn("age_group_45-55", feats.columns)
        # Check Asian Indian BMI categories
        self.assertIn("bmi_category", feats.columns)
        self.assertIn("bmi_category_code", feats.columns)
        # Check HbA1c tiers
        self.assertIn("hba1c_tier", feats.columns)
        self.assertIn("hba1c_tier_code", feats.columns)
        # Check Comorbidity & medication scores
        self.assertIn("comorbidity_count", feats.columns)
        self.assertIn("comorbidity_severity_score", feats.columns)
        self.assertIn("medication_complexity_score", feats.columns)
        # Check composite scores
        self.assertIn("lifestyle_composite_score", feats.columns)
        self.assertIn("cardiovascular_risk_composite", feats.columns)
        self.assertIn("egfr_stage", feats.columns)
        # Check Indian-specific factors
        self.assertIn("carb_protein_imbalance", feats.columns)
        self.assertIn("south_asian_phenotype_score", feats.columns)

    def test_temporal_feature_extraction(self):
        """Verifies circadian, momentum, volatility, and recovery features."""
        dates = pd.date_range("2026-03-01", periods=144, freq="5min")
        cgm_sample = pd.DataFrame({
            "patient_id": ["IND-T2D-001"] * 144,
            "timestamp": dates,
            "glucose": 135.0 + 20.0 * np.sin(np.linspace(0, 2 * np.pi, 144)),
            "heart_rate": 72.0 + 10.0 * np.random.rand(144),
            "steps": np.random.poisson(20, 144),
        })
        temporal_feats = extract_temporal_features(cgm_sample)

        self.assertEqual(len(temporal_feats), 144)
        # Circadian
        self.assertIn("sin_time_of_day", temporal_feats.columns)
        self.assertIn("cos_time_of_day", temporal_feats.columns)
        # Momentum & Volatility
        self.assertIn("glucose_momentum_fast", temporal_feats.columns)
        self.assertIn("glucose_volatility_index", temporal_feats.columns)
        # Multi-scale
        self.assertIn("glucose_mean_15m", temporal_feats.columns)
        self.assertIn("glucose_mean_1h", temporal_feats.columns)
        # Autonomic recovery
        self.assertIn("hr_recovery_delta", temporal_feats.columns)


class TestFeatureStore(unittest.TestCase):
    """Test suite for FeatureStore registry, scaling, and persistence."""

    def test_fit_transform_and_save_load(self):
        """Verifies leakage-free scaling and artifact serialization."""
        store = FeatureStore(store_dir=PROJECT_ROOT / "data" / "processed")
        ehr = generate_synthetic_ehr_cohort(num_patients=6, seed=202)
        dates = pd.date_range("2026-03-01", periods=72, freq="5min")
        cgm_records = []
        for pid in ehr["patient_id"]:
            cgm_records.append(pd.DataFrame({
                "patient_id": [pid] * 72,
                "timestamp": dates,
                "glucose": 120.0 + 15.0 * np.random.rand(72),
                "heart_rate": 75.0,
                "steps": 10.0,
            }))
        cgm = pd.concat(cgm_records, axis=0).reset_index(drop=True)

        static_df, temp_df = store.build_features(ehr, cgm)
        train_pids = list(ehr["patient_id"][:4])

        store.fit(static_df, temp_df, train_pids)
        self.assertTrue(store.is_fitted)

        # Scale features
        scaled_s = store.transform_static(static_df)
        scaled_t = store.transform_temporal(temp_df)
        self.assertEqual(scaled_s.shape[0], len(ehr))
        self.assertEqual(scaled_t.shape[0], len(cgm))

        # Test persistence
        save_path = store.save()
        self.assertTrue(save_path.exists())
        loaded = FeatureStore.load(save_path)
        self.assertTrue(loaded.is_fitted)
        self.assertEqual(len(loaded.dynamic_feature_cols), len(store.dynamic_feature_cols))


class TestDataFusion(unittest.TestCase):
    """Test suite for Multimodal Data Fusion and patient-level splitting."""

    def test_merge_and_sequence_creation(self):
        """Verifies broadcast merge and tensor generation."""
        ehr = generate_synthetic_ehr_cohort(num_patients=4, seed=303)
        cgm = CGMProcessor.generate_synthetic_cgm(num_patients=4, days=1, seed=303)
        processor = CGMProcessor()
        processor.load(cgm)
        processed_cgm = processor.process_all(save=False)

        fusion = DataFusion(seq_length=12, prediction_horizon=12)
        fusion.load_ehr_features(ehr)
        fusion.load_cgm_features(processed_cgm)

        merged = fusion.merge()
        self.assertGreater(len(merged), 0)
        self.assertEqual(merged["patient_id"].nunique(), 4)

        # Test sequence generation
        X_temp, X_stat, y, pids, ts = fusion.create_sequences(stride=1)
        self.assertEqual(len(X_temp), len(y))
        self.assertEqual(X_temp.shape[1], 12)  # seq_length
        self.assertEqual(X_stat.shape[0], len(y))

        # Test patient-level split (no cross-patient leakage)
        splits = fusion.split_data(test_size=0.25, val_size=0.25, stride=2)
        train_p = set(splits["train"]["cohort_patient_ids"])
        val_p = set(splits["val"]["cohort_patient_ids"])
        test_p = set(splits["test"]["cohort_patient_ids"])

        # Disjoint sets guarantee zero leakage
        self.assertEqual(len(train_p.intersection(val_p)), 0)
        self.assertEqual(len(train_p.intersection(test_p)), 0)
        self.assertEqual(len(val_p.intersection(test_p)), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
