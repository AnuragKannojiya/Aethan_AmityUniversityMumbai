"""Unit and Integration Tests for GlucoTwin Data Generation & EHR Pipeline"""

import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_synthetic import (
    generate_ehr_data,
    generate_medications_schedule,
    simulate_patient_cgm,
    extract_patient_events,
    generate_all_data
)
from src.data_pipeline.ehr_processor import (
    EHRProcessor,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    KNOWN_COMORBIDITIES,
    RISK_TIERS
)


class TestSyntheticDataGeneration(unittest.TestCase):
    """Test suite for synthetic data generation modules."""

    @classmethod
    def setUpClass(cls):
        cls.ehr_df = generate_ehr_data(n_patients=10, seed=123)

    def test_ehr_patient_count_and_columns(self):
        """Test EHR generation schema and count."""
        expected_cols = [
            "patient_id", "name", "age", "gender", "city", "bmi",
            "weight_kg", "height_cm", "hba1c", "fasting_glucose",
            "blood_pressure_systolic", "blood_pressure_diastolic",
            "cholesterol_total", "cholesterol_ldl", "cholesterol_hdl",
            "triglycerides", "creatinine", "egfr", "diabetes_duration_years",
            "medication", "comorbidities", "family_history_diabetes",
            "smoking_status", "physical_activity_level", "diet_type",
            "stress_level", "sleep_hours_avg", "diagnosis_date", "last_visit_date"
        ]
        self.assertEqual(len(self.ehr_df), 10)
        for col in expected_cols:
            self.assertIn(col, self.ehr_df.columns, f"Missing expected column: {col}")

    def test_ehr_clinical_plausibility(self):
        """Test physiological bounds and correlations in generated EHR data."""
        self.assertTrue((self.ehr_df["age"] >= 35).all() and (self.ehr_df["age"] <= 75).all())
        self.assertTrue((self.ehr_df["hba1c"] >= 5.5).all() and (self.ehr_df["hba1c"] <= 12.0).all())
        self.assertTrue((self.ehr_df["fasting_glucose"] >= 100.0).all() and (self.ehr_df["fasting_glucose"] <= 300.0).all())
        self.assertTrue((self.ehr_df["bmi"] >= 22.0).all() and (self.ehr_df["bmi"] <= 42.0).all())
        self.assertTrue((self.ehr_df["blood_pressure_systolic"] >= 110).all())
        self.assertTrue((self.ehr_df["blood_pressure_diastolic"] >= 70).all())

    def test_medication_schedules(self):
        """Test medication schedule generation."""
        meds = generate_medications_schedule(self.ehr_df, seed=42)
        self.assertGreater(len(meds), 0)
        self.assertIn("patient_id", meds.columns)
        self.assertIn("dosage", meds.columns)
        self.assertIn("frequency", meds.columns)
        self.assertIn("timing_hours", meds.columns)
        self.assertIn("adherence_rate", meds.columns)
        self.assertTrue((meds["adherence_rate"] >= 0.70).all() and (meds["adherence_rate"] <= 1.0).all())

    def test_cgm_simulation(self):
        """Test CGM simulation for single patient over 2 days."""
        patient_row = self.ehr_df.iloc[0]
        cgm_df, events = simulate_patient_cgm(patient_row, days=2, seed=999)
        
        expected_readings = 2 * 288  # 576
        self.assertEqual(len(cgm_df), expected_readings)
        
        required_cols = [
            "patient_id", "timestamp", "glucose_mg_dl", "heart_rate_bpm",
            "steps_last_5min", "sleep_stage", "skin_temperature_c",
            "stress_index", "activity_type"
        ]
        for col in required_cols:
            self.assertIn(col, cgm_df.columns)

        # Bounds checks
        self.assertTrue((cgm_df["glucose_mg_dl"] >= 40.0).all() and (cgm_df["glucose_mg_dl"] <= 400.0).all())
        self.assertTrue((cgm_df["heart_rate_bpm"] >= 40).all() and (cgm_df["heart_rate_bpm"] <= 150).all())
        self.assertTrue((cgm_df["steps_last_5min"] >= 0).all() and (cgm_df["steps_last_5min"] <= 500).all())
        self.assertTrue((cgm_df["skin_temperature_c"] >= 33.0).all() and (cgm_df["skin_temperature_c"] <= 38.0).all())
        self.assertTrue((cgm_df["stress_index"] >= 0).all() and (cgm_df["stress_index"] <= 100).all())


class TestEHRProcessor(unittest.TestCase):
    """Test suite for EHRProcessor class."""

    @classmethod
    def setUpClass(cls):
        cls.processor = EHRProcessor()
        cls.processed_df, cls.features = cls.processor.process_all()

    def test_processing_shapes(self):
        """Test output dimensions and feature counts."""
        self.assertEqual(len(self.processed_df), 200)
        self.assertEqual(self.features.shape[0], 200)
        self.assertGreater(self.features.shape[1], 20)
        self.assertEqual(self.features.shape[1], len(self.processor.feature_names))

    def test_risk_stratification(self):
        """Test risk stratification categories."""
        self.assertIn("risk_stratification", self.processed_df.columns)
        self.assertIn("risk_score", self.processed_df.columns)
        self.assertIn("risk_class", self.processed_df.columns)

        found_tiers = set(self.processed_df["risk_stratification"].unique())
        for tier in found_tiers:
            self.assertIn(tier, RISK_TIERS)

    def test_comorbidity_encoding(self):
        """Test binary comorbidity multi-label decomposition."""
        for comorb in KNOWN_COMORBIDITIES:
            col = f"comorbidity_{comorb}"
            self.assertIn(col, self.processed_df.columns)
            self.assertTrue(set(self.processed_df[col].unique()).issubset({0, 1}))

    def test_medication_one_hot(self):
        """Test medication one hot features."""
        med_cols = [c for c in self.processed_df.columns if c.startswith("med_")]
        self.assertGreater(len(med_cols), 0)
        # Check that row-sum of one hot medication is 1.0
        row_sums = self.processed_df[med_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0)

    def test_artifacts_saved_and_loaded(self):
        """Test serialization and reloading of artifacts."""
        proc_reloaded = EHRProcessor()
        proc_reloaded.load_artifacts()
        self.assertTrue(proc_reloaded.is_fitted)
        self.assertEqual(len(proc_reloaded.feature_names), len(self.processor.feature_names))

    def test_transform_single(self):
        """Test single patient inference transform."""
        sample_patient = {
            "patient_id": "P999",
            "name": "Test Patient",
            "age": 52,
            "gender": "M",
            "city": "Mumbai",
            "bmi": 29.5,
            "weight_kg": 78.0,
            "height_cm": 168.0,
            "hba1c": 8.4,
            "fasting_glucose": 172.0,
            "blood_pressure_systolic": 138,
            "blood_pressure_diastolic": 88,
            "cholesterol_total": 210,
            "cholesterol_ldl": 135,
            "cholesterol_hdl": 38,
            "triglycerides": 210,
            "creatinine": 1.1,
            "egfr": 78,
            "diabetes_duration_years": 7.5,
            "medication": "Metformin + Glimepiride",
            "comorbidities": "Hypertension, Dyslipidemia",
            "family_history_diabetes": True,
            "smoking_status": "Former",
            "physical_activity_level": "Light",
            "diet_type": "Vegetarian",
            "stress_level": "Moderate",
            "sleep_hours_avg": 6.5,
            "diagnosis_date": "2018-07-01",
            "last_visit_date": "2025-11-20"
        }
        vec = self.processor.transform_single(sample_patient)
        self.assertEqual(vec.shape, (1, self.features.shape[1]))
        self.assertEqual(vec.dtype, np.float32)
        self.assertFalse(np.isnan(vec).any())


if __name__ == "__main__":
    unittest.main()
