"""
Unit & Integration Tests for GlucoTwin ML Models and Digital Twin Engine
========================================================================
Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026
"""

import unittest
from pathlib import Path
import numpy as np
import torch

from src.models.lstm_model import GlucoseLSTM
from src.models.xgboost_model import GlucoseXGBoost
from src.models.ensemble import GlucoTwinEnsemble
from src.digital_twin.twin_engine import DigitalTwin
from src.digital_twin.risk_calculator import RiskCalculator
from src.digital_twin.simulator import PatientSimulator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestGlucoTwinModelsAndEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models_dir = PROJECT_ROOT / "data" / "processed" / "models"
        cls.batch_size = 4
        cls.seq_len = 24
        cls.temporal_dim = 54
        cls.static_dim = 53

        # Dummy inputs for testing
        cls.x_temp = np.random.randn(cls.batch_size, cls.seq_len, cls.temporal_dim).astype(np.float32)
        cls.x_stat = np.random.randn(cls.batch_size, cls.static_dim).astype(np.float32)

    def test_lstm_forward_and_attention(self):
        """Verify LSTM model forward pass, output shape, and attention extraction."""
        model = GlucoseLSTM(
            temporal_dim=self.temporal_dim,
            static_dim=self.static_dim,
            hidden_dim=32,
            num_lstm_layers=1,
            dropout=0.1,
        )
        model.eval()

        t_temp = torch.as_tensor(self.x_temp)
        t_stat = torch.as_tensor(self.x_stat)

        with torch.no_grad():
            out = model(t_temp, t_stat)

        self.assertEqual(out.shape, (self.batch_size, 1))
        # Probabilities must be strictly bounded in [0, 1]
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

        # Check attention weights
        attn = model.get_attention_weights(t_temp, t_stat)
        self.assertEqual(attn.shape, (self.batch_size, self.seq_len))
        self.assertTrue(np.allclose(attn.sum(axis=1), 1.0, atol=1e-4))

    def test_saved_model_artifacts_and_ensemble(self):
        """Verify that trained model artifacts exist and load into the GlucoTwinEnsemble."""
        if (self.models_dir / "lstm_model.pt").exists() and (self.models_dir / "xgboost_model.json").exists():
            ensemble = GlucoTwinEnsemble.load(self.models_dir)
            self.assertIsNotNone(ensemble.lstm_model)
            self.assertIsNotNone(ensemble.xgb_model)

            # Predict with ensemble probabilities
            probs = ensemble.predict_proba(self.x_temp, self.x_stat)
            self.assertEqual(len(probs), self.batch_size)
            self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

            # Predict with confidence dict
            single_temp = self.x_temp[0]
            single_stat = self.x_stat[0]
            conf_dict = ensemble.predict_with_confidence(single_temp, single_stat)
            self.assertIn("spike_probability", conf_dict)
            self.assertIn("confidence", conf_dict)
            self.assertIn("prediction", conf_dict)
            self.assertTrue(0.0 <= conf_dict["spike_probability"] <= 1.0)
            self.assertTrue(0.5 <= conf_dict["confidence"] <= 1.0)

    def test_risk_calculator(self):
        """Verify RiskCalculator clinical thresholds and recommendation generation."""
        calc = RiskCalculator()

        # Critical risk scenario
        crit_res = calc.calculate_composite_risk(
            ehr_features={"hba1c": 10.5, "medication": "Metformin + Glimepiride"},
            cgm_features={"glucose": 265.0, "rate_of_change": 2.5, "steps": 50},
            model_prediction=0.92,
        )
        self.assertEqual(crit_res["risk_level"], "Critical")
        self.assertGreater(crit_res["composite_score"], 80.0)

        recs = calc.generate_recommendations(crit_res["risk_level"], crit_res["risk_factors"])
        self.assertGreater(len(recs), 0)

        # Low risk scenario
        low_res = calc.calculate_composite_risk(
            ehr_features={"hba1c": 6.2, "medication": "Metformin"},
            cgm_features={"glucose": 110.0, "rate_of_change": -0.2, "steps": 300},
            model_prediction=0.08,
        )
        self.assertEqual(low_res["risk_level"], "Low")
        self.assertLess(low_res["composite_score"], 40.0)

    def test_patient_simulator(self):
        """Verify PatientSimulator counterfactual interventions."""
        sim = PatientSimulator()
        baseline_trajectory = [
            {"time": f"2026-10-15T12:{m:02d}:00Z", "glucose": g}
            for m, g in [(10, 180.0), (20, 195.0), (30, 215.0), (40, 230.0), (50, 240.0), (60, 235.0)]
        ]

        # Test exercise intervention (GLUT4 non-insulin mediated glucose uptake)
        walk_sim = sim.generate_counterfactual(
            baseline_trajectory=baseline_trajectory,
            intervention="walk",
            intensity="moderate",
            patient_profile={"hba1c": 8.0, "bmi": 27.0},
        )
        self.assertEqual(len(walk_sim["intervention_trajectory"]), len(baseline_trajectory))
        self.assertIn("expected_improvement", walk_sim)
        self.assertGreater(walk_sim["peak_reduction_mg_dl"], 10.0)

        # Test medication intervention
        med_sim = sim.generate_counterfactual(
            baseline_trajectory=baseline_trajectory,
            intervention="medication",
            intensity="moderate",
            patient_profile={"hba1c": 8.0, "bmi": 27.0},
        )
        self.assertGreater(med_sim["peak_reduction_mg_dl"], 5.0)

    def test_digital_twin_engine_state_and_prediction(self):
        """Verify DigitalTwin initialization, telemetry ingestion, and trajectory forecasting."""
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
            "comorbidities": "Hypertension, Dyslipidemia",
        }

        twin = DigitalTwin(patient_id="P001", ehr_data=patient_profile)

        # Ingest streaming readings
        simulated_readings = [
            {"timestamp": "2026-10-15T12:00:00Z", "glucose": 160.0, "heart_rate": 78, "steps": 120, "activity_type": "Walking"},
            {"timestamp": "2026-10-15T12:05:00Z", "glucose": 168.0, "heart_rate": 82, "steps": 180, "activity_type": "Walking"},
            {"timestamp": "2026-10-15T12:10:00Z", "glucose": 178.0, "heart_rate": 85, "steps": 210, "activity_type": "Walking"},
        ]
        twin.update_state(simulated_readings)

        # Check patient summary
        summary = twin.get_patient_summary()
        self.assertEqual(summary["patient_id"], "P001")
        self.assertEqual(summary["last_glucose"], 178.0)
        self.assertEqual(summary["name"], "Rajesh Sharma")

        # Check trajectory forecasting
        trajectory = twin.predict_glucose_trajectory(hours_ahead=2)
        self.assertGreaterEqual(len(trajectory), 8)
        self.assertIn("time", trajectory[0])
        self.assertIn("glucose", trajectory[0])

        # Check clinical risk assessment
        risk = twin.assess_risk()
        self.assertIn("risk_level", risk)
        self.assertIn("spike_probability", risk)
        self.assertIn("recommended_actions", risk)

        # Check alerts
        alerts = twin.get_alerts()
        self.assertIsInstance(alerts, list)

        # Check counterfactual simulation
        sim_res = twin.simulate_intervention(intervention_type="walk")
        self.assertIn("baseline_trajectory", sim_res)
        self.assertIn("intervention_trajectory", sim_res)


if __name__ == "__main__":
    unittest.main()
