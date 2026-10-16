"""GlucoTwin: Tabular Gradient Boosting Model for Glucose Spike Prediction"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger("glucotwin.models.xgboost")


class GlucoseXGBoost:
    """Gradient Boosted Decision Tree pipeline for multimodal glucose spike prediction."""

    def __init__(
        self,
        n_estimators: int = 150,
        max_depth: int = 6,
        learning_rate: float = 0.08,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[Union[float, str]] = "auto",
        tree_method: str = "hist",
        random_state: int = 42,
        n_jobs: int = -1,
        feature_names: Optional[List[str]] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.feature_names: List[str] = feature_names or []

        self.model: Optional[xgb.XGBClassifier] = None
        self.is_fitted: bool = False
        self.effective_scale_pos_weight: float = 1.0

    # Feature Flattening & Summary Statistics

    @staticmethod
    def extract_tabular_features(
        X_temporal: np.ndarray,
        X_static: np.ndarray,
        dynamic_names: Optional[List[str]] = None,
        static_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Transforms 3D temporal sequences (N, 24, 54) and static features (N, 46)"""
        if X_temporal.ndim == 2:
            X_temporal = np.expand_dims(X_temporal, axis=0)
        if X_static.ndim == 1:
            X_static = np.expand_dims(X_static, axis=0)

        n_samples, seq_len, n_dyn = X_temporal.shape
        n_stat = X_static.shape[1]

        # 1. Last timestep (current state)
        feat_last = X_temporal[:, -1, :]

        # 2. 2-Hour summary statistics
        feat_mean_2h = X_temporal.mean(axis=1)
        feat_std_2h = X_temporal.std(axis=1)
        feat_min_2h = X_temporal.min(axis=1)
        feat_max_2h = X_temporal.max(axis=1)
        feat_delta_2h = X_temporal[:, -1, :] - X_temporal[:, 0, :]

        # 3. 30-Minute short-term kinetic window (last 6 steps at 5-min cadence)
        short_window = X_temporal[:, -6:, :] if seq_len >= 6 else X_temporal
        feat_mean_30m = short_window.mean(axis=1)
        feat_delta_30m = X_temporal[:, -1, :] - X_temporal[:, -min(seq_len, 6), :]

        # Concatenate all tabular blocks along feature axis
        X_flat = np.concatenate(
            [
                feat_last,
                feat_mean_2h,
                feat_std_2h,
                feat_min_2h,
                feat_max_2h,
                feat_delta_2h,
                feat_mean_30m,
                feat_delta_30m,
                X_static,
            ],
            axis=1,
            dtype=np.float32,
        )

        # Build feature names
        dyn_cols = dynamic_names or [f"dyn_{i}" for i in range(n_dyn)]
        stat_cols = static_names or [f"stat_{i}" for i in range(n_stat)]

        feature_names: List[str] = []
        for c in dyn_cols:
            feature_names.append(f"{c}_last")
        for c in dyn_cols:
            feature_names.append(f"{c}_mean_2h")
        for c in dyn_cols:
            feature_names.append(f"{c}_std_2h")
        for c in dyn_cols:
            feature_names.append(f"{c}_min_2h")
        for c in dyn_cols:
            feature_names.append(f"{c}_max_2h")
        for c in dyn_cols:
            feature_names.append(f"{c}_delta_2h")
        for c in dyn_cols:
            feature_names.append(f"{c}_mean_30m")
        for c in dyn_cols:
            feature_names.append(f"{c}_delta_30m")
        for c in stat_cols:
            feature_names.append(f"{c}")

        return X_flat, feature_names

    def _prepare_input(
        self,
        X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Helper to unpack tuple of (X_temporal, X_static) or return 2D array."""
        if isinstance(X, (tuple, list)) and len(X) == 2:
            x_temp, x_stat = X[0], X[1]
            if x_temp.ndim >= 2 and x_stat.ndim >= 1:
                flat_x, names = self.extract_tabular_features(x_temp, x_stat)
                if not self.feature_names:
                    self.feature_names = names
                return flat_x
        return np.asarray(X, dtype=np.float32)

    # Training & Evaluation

    def train(
        self,
        X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        y: np.ndarray,
        eval_set: Optional[List[Tuple[Any, np.ndarray]]] = None,
        early_stopping_rounds: int = 15,
        verbose: bool = False,
    ) -> "GlucoseXGBoost":
        """Train the XGBoost classifier with class weighting and optional early stopping."""
        X_train_flat = self._prepare_input(X)
        y_train = np.asarray(y, dtype=np.int32).ravel()

        # Compute imbalanced class weighting
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)

        if self.scale_pos_weight == "auto":
            # Ratio of negative to positive samples
            weight = float(n_neg) / float(max(1, n_pos))
            self.effective_scale_pos_weight = weight
        elif isinstance(self.scale_pos_weight, (int, float)):
            self.effective_scale_pos_weight = float(self.scale_pos_weight)
        else:
            self.effective_scale_pos_weight = 1.0

        logger.info(
            "Training XGBoost: samples=%d, features=%d, pos=%d (%.1f%%), neg=%d, scale_pos_weight=%.4f",
            X_train_flat.shape[0],
            X_train_flat.shape[1],
            n_pos,
            (n_pos / max(1, len(y_train))) * 100,
            n_neg,
            self.effective_scale_pos_weight,
        )

        prepared_eval_set = None
        if eval_set:
            prepared_eval_set = []
            for ev_x, ev_y in eval_set:
                flat_ev_x = self._prepare_input(ev_x)
                flat_ev_y = np.asarray(ev_y, dtype=np.int32).ravel()
                prepared_eval_set.append((flat_ev_x, flat_ev_y))

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=self.effective_scale_pos_weight,
            tree_method=self.tree_method,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            eval_metric="logloss",
            early_stopping_rounds=early_stopping_rounds if prepared_eval_set else None,
        )

        self.model.fit(
            X_train_flat,
            y_train,
            eval_set=prepared_eval_set,
            verbose=verbose,
        )

        self.is_fitted = True
        logger.info("XGBoost training complete. Best iteration: %s", getattr(self.model, "best_iteration", "N/A"))
        return self

    # Inference

    def predict_proba(
        self,
        X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        return_2d: bool = False,
    ) -> np.ndarray:
        """Compute predicted probability of glucose spike (>180 mg/dL within 2 hours)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("GlucoseXGBoost model is not fitted yet.")

        X_flat = self._prepare_input(X)
        proba_2d = self.model.predict_proba(X_flat)

        if return_2d:
            return proba_2d
        # Return probability of spike (class 1)
        p1 = proba_2d[:, 1]
        if len(p1) == 1 and not isinstance(X, np.ndarray):
            return float(p1[0])
        return p1

    def predict(
        self,
        X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Binary prediction of glucose spike event."""
        probs = self.predict_proba(X, return_2d=False)
        if np.isscalar(probs):
            return int(probs >= threshold)
        return (probs >= threshold).astype(int)

    # Explainability & Feature Importance

    def get_feature_importance(
        self,
        top_n: int = 20,
        importance_type: str = "gain",
    ) -> pd.DataFrame:
        """Extract ranked feature importances for clinical interpretability."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is not fitted yet.")

        booster = self.model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)

        # Map internal feature identifiers (f0, f1, ...) to descriptive names if available
        records = []
        total_features = len(self.feature_names) if self.feature_names else self.model.n_features_in_

        for i in range(total_features):
            default_key = f"f{i}"
            val = score_dict.get(default_key, 0.0)
            name = self.feature_names[i] if i < len(self.feature_names) else default_key
            records.append({"feature": name, "importance": float(val)})

        df = pd.DataFrame(records).sort_values("importance", ascending=False).reset_index(drop=True)
        total_gain = df["importance"].sum()
        df["relative_importance"] = (df["importance"] / (total_gain if total_gain > 0 else 1.0)) * 100.0
        df["rank"] = df.index + 1

        return df.head(top_n)

    # Persistence

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist XGBoost model and feature schema metadata."""
        target_path = Path(filepath)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save unfitted model.")

        # Save model JSON
        self.model.save_model(str(target_path))

        # Save auxiliary metadata (feature names, hyperparameters)
        meta_path = target_path.with_suffix(".json.meta")
        meta = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": self.effective_scale_pos_weight,
            "tree_method": self.tree_method,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved GlucoseXGBoost model to %s and metadata to %s", target_path, meta_path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "GlucoseXGBoost":
        """Load trained GlucoseXGBoost estimator from disk."""
        target_path = Path(filepath)
        if not target_path.exists():
            raise FileNotFoundError(f"XGBoost model file not found at {target_path}")

        meta_path = target_path.with_suffix(".json.meta")
        meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        instance = cls(
            n_estimators=meta.get("n_estimators", 150),
            max_depth=meta.get("max_depth", 6),
            learning_rate=meta.get("learning_rate", 0.08),
            subsample=meta.get("subsample", 0.8),
            colsample_bytree=meta.get("colsample_bytree", 0.8),
            scale_pos_weight=meta.get("scale_pos_weight", 1.0),
            tree_method=meta.get("tree_method", "hist"),
            random_state=meta.get("random_state", 42),
            feature_names=meta.get("feature_names", []),
        )

        instance.model = xgb.XGBClassifier()
        instance.model.load_model(str(target_path))
        instance.is_fitted = True
        instance.effective_scale_pos_weight = meta.get("scale_pos_weight", 1.0)
        logger.info("Loaded GlucoseXGBoost model from %s with %d feature names", target_path, len(instance.feature_names))
        return instance
