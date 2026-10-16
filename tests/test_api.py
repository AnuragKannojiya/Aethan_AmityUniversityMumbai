"""
Unit Tests for GlucoTwin API Endpoints and Service Integration
=============================================================
Validates all REST API routes, schemas, data serialization, and
Digital Twin forecasting/simulation pipelines.

Author: Team Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import (
    AlertResponse,
    DashboardData,
    GlucoseReading,
    GlucoseStats,
    HealthResponse,
    InterventionRequest,
    InterventionResponse,
    PatientDetail,
    PatientSummary,
    PredictionResponse,
    SystemOverviewStats,
)
from src.api.service import GlucoTwinService
from src.api.routes import FlexibleInterventionRequest


class TestGlucoTwinAPI(unittest.TestCase):
    """Test suite verifying GlucoTwin Service and API endpoints."""

    @classmethod
    def setUpClass(cls):
        """Initialize service singleton once for test suite."""
        cls.service = GlucoTwinService()
        cls.service.initialize()

    def test_service_initialization(self):
        """Test service successfully loaded patients and CGM telemetry."""
        self.assertTrue(self.service.is_initialized)
        unique_patients = len({p.get("patient_id") for p in self.service.patients_dict.values()})
        self.assertEqual(unique_patients, 200)
        self.assertTrue(len(self.service.cgm_readings_dict) >= 200)

    def test_get_all_patients(self):
        """Test listing patient summaries with pagination and filtering."""
        patients = self.service.get_all_patients(limit=10, offset=0)
        self.assertEqual(len(patients), 10)
        for p in patients:
            self.assertIsInstance(p, PatientSummary)
            self.assertTrue(p.patient_id.startswith("P") or p.patient_id.startswith("IND-T2D-"))
            self.assertIn(p.risk_level, ["Low", "Moderate", "High", "Critical"])
            self.assertGreater(p.last_glucose, 0)

    def test_patient_filter_by_risk(self):
        """Test filtering patients by risk level."""
        high_risk_patients = self.service.get_all_patients(risk_level="High", limit=20)
        self.assertTrue(len(high_risk_patients) > 0)
        for p in high_risk_patients:
            self.assertEqual(p.risk_level.lower(), "high")

    def test_get_patient_detail(self):
        """Test retrieving detailed clinical patient profile."""
        patient = self.service.get_patient("P001")
        self.assertIsNotNone(patient)
        self.assertIsInstance(patient, PatientDetail)
        self.assertEqual(patient.patient_id, "P001")
        self.assertIsNotNone(patient.name)
        self.assertIsNotNone(patient.hba1c)
        self.assertIsNotNone(patient.blood_pressure_systolic)

    def test_dual_alias_resolution(self):
        """Test that both P001 and IND-T2D-001 resolve to the same patient."""
        p_raw = self.service.get_patient("P001")
        p_alias = self.service.get_patient("IND-T2D-001")
        self.assertIsNotNone(p_raw)
        self.assertIsNotNone(p_alias)
        self.assertEqual(p_raw.name, p_alias.name)

    def test_get_recent_readings(self):
        """Test retrieving 24-hour CGM readings."""
        readings = self.service.get_recent_readings("P001", hours=24)
        self.assertTrue(len(readings) > 0)
        for r in readings:
            self.assertIsInstance(r, GlucoseReading)
            self.assertTrue(20.0 <= r.glucose_mg_dl <= 600.0)
            self.assertIsInstance(r.timestamp, datetime)

    def test_predict_glucose_trajectory(self):
        """Test 2-hour glucose forecasting and spike probability calculation."""
        prediction = self.service.predict_glucose_trajectory("P001")
        self.assertIsNotNone(prediction)
        self.assertIsInstance(prediction, PredictionResponse)
        self.assertEqual(len(prediction.predicted_trajectory), 12)
        self.assertTrue(0.0 <= prediction.spike_probability <= 1.0)
        self.assertIn(prediction.risk_level, ["Low", "Moderate", "High", "Critical"])
        self.assertTrue(len(prediction.risk_factors) > 0)
        self.assertTrue(len(prediction.recommended_actions) > 0)

    def test_simulate_intervention(self):
        """Test 'What-if' counterfactual intervention simulation."""
        req = InterventionRequest(
            patient_id="P001",
            intervention_type="exercise",
            intensity="moderate",
        )
        sim = self.service.simulate_intervention("P001", req)
        self.assertIsNotNone(sim)
        self.assertIsInstance(sim, InterventionResponse)
        self.assertEqual(len(sim.baseline_trajectory), 12)
        self.assertEqual(len(sim.intervention_trajectory), 12)
        self.assertTrue(len(sim.expected_improvement) > 0)

    def test_get_patient_alerts(self):
        """Test alert engine generation."""
        alerts = self.service.get_patient_alerts("P001")
        self.assertIsInstance(alerts, list)
        for a in alerts:
            self.assertIsInstance(a, AlertResponse)
            self.assertIn(a.severity, ["low", "medium", "moderate", "high", "critical"])

    def test_get_glucose_statistics(self):
        """Test International Consensus Time-in-Range (TIR) metrics."""
        stats = self.service.get_glucose_statistics("P001", hours=48)
        self.assertIsNotNone(stats)
        self.assertIsInstance(stats, GlucoseStats)
        self.assertTrue(0.0 <= stats.time_in_range_pct <= 100.0)
        self.assertTrue(0.0 <= stats.time_above_range_pct <= 100.0)
        self.assertTrue(0.0 <= stats.time_below_range_pct <= 100.0)
        self.assertGreater(stats.mean_glucose, 0.0)
        self.assertGreater(stats.gmi, 0.0)

    def test_get_dashboard_data(self):
        """Test consolidated unified dashboard assembler."""
        dashboard = self.service.get_dashboard_data("P001")
        self.assertIsNotNone(dashboard)
        self.assertIsInstance(dashboard, DashboardData)
        self.assertEqual(dashboard.patient.patient_id, "P001")
        self.assertTrue(len(dashboard.recent_readings) > 0)
        self.assertEqual(len(dashboard.prediction.predicted_trajectory), 12)
        self.assertIn("time_in_range_pct", dashboard.glucose_stats)

    def test_system_overview_stats(self):
        """Test cohort-wide analytics calculation."""
        overview = self.service.get_system_overview()
        self.assertIsInstance(overview, SystemOverviewStats)
        self.assertEqual(overview.total_patients, 200)
        self.assertEqual(overview.active_cgm_streams, 200)
        total_risk = (
            overview.low_risk_count
            + overview.moderate_risk_count
            + overview.high_risk_count
            + overview.critical_risk_count
        )
        self.assertEqual(total_risk, 200)
        self.assertGreater(overview.mean_population_glucose, 50.0)
        self.assertGreater(overview.mean_population_hba1c, 5.0)


if __name__ == "__main__":
    unittest.main()
