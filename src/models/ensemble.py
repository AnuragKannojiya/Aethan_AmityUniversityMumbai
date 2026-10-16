"""GlucoTwin: Multimodal Ensemble Engine (LSTM + XGBoost)"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.models.lstm_model import GlucoseLSTM
from src.models.xgboost_model import GlucoseXGBoost

logger = logging.getLogger("glucotwin.models.ensemble")


class GlucoTwinEnsemble:
    """Weighted Multimodal Ensemble combining:"""

    def __init__(
        self,
        lstm_model: Optional[GlucoseLSTM] = None,
        xgb_model: Optional[GlucoseXGBoost] = None,
        lstm_weight: float = 0.6,
        xgb_weight: float = 0.4,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model

        # Normalize weights
        total_weight = lstm_weight + xgb_weight
        if total_weight <= 0:
            self.lstm_weight = 0.6
            self.xgb_weight = 0.4
        else:
            self.lstm_weight = float(lstm_weight) / total_weight
            self.xgb_weight = float(xgb_weight) / total_weight

        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        if self.lstm_model is not None:
            self.lstm_model.to(self.device)
            self.lstm_model.eval()

    # Inference Methods

    def predict_proba(
        self,
        x_temporal: Union[np.ndarray, torch.Tensor],
        x_static: Union[np.ndarray, torch.Tensor],
        x_flat: Optional[np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """Compute weighted ensemble probability of a glucose spike within 2 hours."""
        is_single = (isinstance(x_temporal, np.ndarray) and x_temporal.ndim == 2) or (
            isinstance(x_temporal, torch.Tensor) and x_temporal.ndim == 2
        )

        p_lstm = None
        p_xgb = None

        # 1. LSTM Prediction
        if self.lstm_model is not None:
            p_lstm = self.lstm_model.predict_proba(x_temporal, x_static, device=self.device)
            p_lstm = np.atleast_1d(p_lstm)

        # 2. XGBoost Prediction
        if self.xgb_model is not None:
            if x_flat is not None:
                p_xgb = self.xgb_model.predict_proba(x_flat, return_2d=False)
            else:
                xt_np = (
                    x_temporal.detach().cpu().numpy()
                    if isinstance(x_temporal, torch.Tensor)
                    else np.asarray(x_temporal)
                )
                xs_np = (
                    x_static.detach().cpu().numpy()
                    if isinstance(x_static, torch.Tensor)
                    else np.asarray(x_static)
                )
                p_xgb = self.xgb_model.predict_proba((xt_np, xs_np), return_2d=False)
            p_xgb = np.atleast_1d(p_xgb)

        # 3. Ensemble Fusion
        if p_lstm is not None and p_xgb is not None:
            p_ensemble = self.lstm_weight * p_lstm + self.xgb_weight * p_xgb
        elif p_lstm is not None:
            p_ensemble = p_lstm
        elif p_xgb is not None:
            p_ensemble = p_xgb
        else:
            raise RuntimeError("Neither LSTM nor XGBoost model is loaded into GlucoTwinEnsemble.")

        p_ensemble = np.clip(p_ensemble, 0.0, 1.0)
        if is_single or len(p_ensemble) == 1:
            return float(p_ensemble[0])
        return p_ensemble

    def predict(
        self,
        x_temporal: Union[np.ndarray, torch.Tensor],
        x_static: Union[np.ndarray, torch.Tensor],
        x_flat: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Union[int, np.ndarray]:
        """Binary spike forecast (0: No spike <=180 mg/dL, 1: Spike >180 mg/dL)."""
        probs = self.predict_proba(x_temporal, x_static, x_flat=x_flat)
        if np.isscalar(probs):
            return int(probs >= threshold)
        return (probs >= threshold).astype(int)

    # Confidence & Concordance Scoring

    def predict_with_confidence(
        self,
        x_temporal: Union[np.ndarray, torch.Tensor],
        x_static: Union[np.ndarray, torch.Tensor],
        x_flat: Optional[np.ndarray] = None,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Performs joint prediction, model concordance verification, and uncertainty estimation."""
        xt_np = (
            x_temporal.detach().cpu().numpy()
            if isinstance(x_temporal, torch.Tensor)
            else np.asarray(x_temporal)
        )
        xs_np = (
            x_static.detach().cpu().numpy()
            if isinstance(x_static, torch.Tensor)
            else np.asarray(x_static)
        )

        p_lstm = (
            float(np.atleast_1d(self.lstm_model.predict_proba(x_temporal, x_static, device=self.device))[0])
            if self.lstm_model is not None
            else 0.5
        )

        p_xgb = (
            float(np.atleast_1d(self.xgb_model.predict_proba((xt_np, xs_np), return_2d=False))[0])
            if self.xgb_model is not None
            else p_lstm
        )

        if self.lstm_model is not None and self.xgb_model is not None:
            p_ensemble = float(self.lstm_weight * p_lstm + self.xgb_weight * p_xgb)
        elif self.lstm_model is not None:
            p_ensemble = p_lstm
        else:
            p_ensemble = p_xgb

        p_ensemble = float(np.clip(p_ensemble, 0.01, 0.99))
        pred = int(p_ensemble >= threshold)
        agreement = (p_lstm >= threshold) == (p_xgb >= threshold)

        certainty = 2.0 * abs(p_ensemble - 0.5)
        model_concordance = 1.0 - abs(p_lstm - p_xgb)
        confidence = float(np.clip(0.50 + 0.49 * (0.65 * certainty + 0.35 * model_concordance), 0.50, 0.99))

        # Extract temporal attention weights
        attn_weights: List[float] = []
        if self.lstm_model is not None:
            raw_attn = self.lstm_model.get_attention_weights(xt_np, xs_np, device=self.device)
            if raw_attn.ndim > 1:
                raw_attn = raw_attn[0]
            attn_weights = [round(float(w), 4) for w in raw_attn]
        else:
            attn_weights = [1.0 / 24.0] * 24

        return {
            "spike_probability": round(p_ensemble, 4),
            "prediction": pred,
            "confidence": round(confidence, 4),
            "lstm_probability": round(p_lstm, 4),
            "xgb_probability": round(p_xgb, 4),
            "model_agreement": bool(agreement),
            "attention_weights": attn_weights,
        }

    # Clinical & Multimodal Explainability

    def explain_prediction(
        self,
        x_temporal: Union[np.ndarray, torch.Tensor],
        x_static: Union[np.ndarray, torch.Tensor],
        x_flat: Optional[np.ndarray] = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Generate clinically rich explanations combining:"""
        res = self.predict_with_confidence(x_temporal, x_static, x_flat=x_flat)
        spike_prob = res["spike_probability"]
        attn = res["attention_weights"]

        # Risk level categorization
        if spike_prob >= 0.80:
            risk_level = "Critical"
        elif spike_prob >= 0.55:
            risk_level = "High"
        elif spike_prob >= 0.35:
            risk_level = "Moderate"
        else:
            risk_level = "Low"

        # Find peak attention period
        # Lookback is 24 timesteps (120 minutes, 5-min intervals)
        peak_idx = int(np.argmax(attn))
        mins_ago = (24 - 1 - peak_idx) * 5
        attention_summary = {
            "peak_step_index": peak_idx,
            "peak_minutes_ago": mins_ago,
            "peak_attention_weight": round(float(attn[peak_idx]), 4),
            "recent_30m_weight_sum": round(float(sum(attn[-6:])), 4),
            "attention_weights": attn,
        }

        # Top XGBoost features
        top_features = []
        if self.xgb_model is not None and self.xgb_model.is_fitted:
            fi_df = self.xgb_model.get_feature_importance(top_n=top_k)
            for _, r in fi_df.iterrows():
                top_features.append({
                    "feature": r["feature"],
                    "importance": round(float(r["importance"]), 4),
                    "relative_importance_pct": round(float(r["relative_importance"]), 2),
                })

        # Clinical explanation strings
        clinical_factors = []
        if spike_prob >= 0.55:
            clinical_factors.append(
                f"High model spike risk ({spike_prob*100:.1f}%) detected for the upcoming 2-hour window."
            )
            if mins_ago <= 30:
                clinical_factors.append(
                    f"Recurrent attention concentrates on recent events (~{mins_ago}m ago) reflecting strong upward glycemic momentum."
                )
            else:
                clinical_factors.append(
                    f"Sustained glycemic elevation over the past {mins_ago} minutes contributes heavily to predicted spike."
                )
            if top_features:
                top_names = [f["feature"] for f in top_features[:3]]
                clinical_factors.append(f"Key statistical drivers: {', '.join(top_names)}.")
        else:
            clinical_factors.append(
                f"Glycemic trajectory is projected stable (Spike risk: {spike_prob*100:.1f}%, Low/Moderate tier)."
            )
            clinical_factors.append("No acute physiological spike surge expected within 2 hours.")

        # Actionable recommendations
        recommendations = []
        if risk_level in ["Critical", "High"]:
            recommendations.append("Initiate 15-minute moderate brisk walk to promote GLUT4 muscle glucose uptake.")
            recommendations.append("Hydrate with 400ml water to assist renal glucose filtration.")
            recommendations.append("Review afternoon/evening anti-diabetic medication adherence.")
            recommendations.append("Defer high-glycemic carbohydrates or sweets over the next 2 hours.")
        elif risk_level == "Moderate":
            recommendations.append("Perform light post-meal movement (500 steps) and monitor CGM trend arrow.")
            recommendations.append("Drink water and maintain standard dietary schedule.")
        else:
            recommendations.append("Maintain optimal routine; glucose is well within target range.")

        return {
            "spike_probability": spike_prob,
            "risk_level": risk_level,
            "confidence": res["confidence"],
            "model_agreement": res["model_agreement"],
            "lstm_probability": res["lstm_probability"],
            "xgb_probability": res["xgb_probability"],
            "attention_summary": attention_summary,
            "top_features": top_features,
            "clinical_factors": clinical_factors,
            "recommendations": recommendations,
        }

    # Persistence

    def save(self, save_dir: Union[str, Path]) -> None:
        """Save ensemble artifacts to destination directory:"""
        dir_path = Path(save_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        if self.lstm_model is not None:
            self.lstm_model.save(dir_path / "lstm_model.pt")

        if self.xgb_model is not None and self.xgb_model.is_fitted:
            self.xgb_model.save(dir_path / "xgboost_model.json")

        config = {
            "lstm_weight": self.lstm_weight,
            "xgb_weight": self.xgb_weight,
            "has_lstm": self.lstm_model is not None,
            "has_xgb": self.xgb_model is not None and self.xgb_model.is_fitted,
        }
        with open(dir_path / "ensemble_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved GlucoTwinEnsemble artifacts to %s", dir_path)

    @classmethod
    def load(
        cls,
        save_dir: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "GlucoTwinEnsemble":
        """Load trained ensemble from artifact directory."""
        dir_path = Path(save_dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"Ensemble directory not found at {dir_path}")

        config_path = dir_path / "ensemble_config.json"
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

        lstm_weight = config.get("lstm_weight", 0.6)
        xgb_weight = config.get("xgb_weight", 0.4)

        lstm_model = None
        lstm_path = dir_path / "lstm_model.pt"
        if lstm_path.exists():
            lstm_model = GlucoseLSTM.load(lstm_path, device=device)

        xgb_model = None
        xgb_path = dir_path / "xgboost_model.json"
        if xgb_path.exists():
            xgb_model = GlucoseXGBoost.load(xgb_path)

        return cls(
            lstm_model=lstm_model,
            xgb_model=xgb_model,
            lstm_weight=lstm_weight,
            xgb_weight=xgb_weight,
            device=device,
        )
