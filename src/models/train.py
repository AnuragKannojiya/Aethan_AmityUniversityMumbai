"""
GlucoTwin: End-to-End Model Training & Evaluation Pipeline
==========================================================
Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026

Trains both the PyTorch multimodal LSTM and XGBoost models on processed T2D cohorts,
constructs the calibrated GlucoTwin ensemble, evaluates performance across clinical
metrics (ROC-AUC, PR-AUC, Sensitivity, Specificity, Brier Score), and persists artifacts.

Usage:
------
    python -m src.models.train
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble import GlucoTwinEnsemble
from src.models.lstm_model import GlucoseLSTM
from src.models.xgboost_model import GlucoseXGBoost

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("glucotwin.train")


def get_device() -> torch.device:
    """Select compute device (Apple Silicon MPS -> CUDA -> CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Using compute device: %s", device)
    return device


def load_dataset(data_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load train, validation, and test splits from .npz files and feature metadata."""
    train_path = data_dir / "train_data.npz"
    val_path = data_dir / "val_data.npz"
    test_path = data_dir / "test_data.npz"
    meta_path = data_dir / "feature_metadata.json"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Processed data files missing in {data_dir}. Ensure train_data.npz, val_data.npz, and test_data.npz exist."
        )

    logger.info("Loading dataset splits from %s...", data_dir)
    train_data = dict(np.load(train_path))
    val_data = dict(np.load(val_path))
    test_data = dict(np.load(test_path))

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    logger.info(
        "Loaded: Train=%d samples, Val=%d samples, Test=%d samples",
        len(train_data["y"]),
        len(val_data["y"]),
        len(test_data["y"]),
    )
    return train_data, val_data, test_data, metadata


# -----------------------------------------------------------------------------
# LSTM Training Loop
# -----------------------------------------------------------------------------


def train_lstm(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    device: torch.device,
    epochs: int = 12,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    patience: int = 4,
    max_train_samples: Optional[int] = None,
) -> GlucoseLSTM:
    """
    Train GlucoseLSTM model with Adam optimizer, Cosine Annealing,
    imbalance class weighting, and early stopping.
    """
    logger.info("=" * 70)
    logger.info("TRAINING PYTORCH GLUCOSE-LSTM WITH TEMPORAL ATTENTION")
    logger.info("=" * 70)

    # Subsample if requested for fast hackathon turnaround
    n_train = len(train_data["y"])
    if max_train_samples and max_train_samples < n_train:
        indices = np.random.RandomState(42).choice(n_train, size=max_train_samples, replace=False)
        xt_train = train_data["X_temporal"][indices]
        xs_train = train_data["X_static"][indices]
        y_train = train_data["y"][indices]
        logger.info("Subsampled training set to %d samples for rapid convergence.", max_train_samples)
    else:
        xt_train = train_data["X_temporal"]
        xs_train = train_data["X_static"]
        y_train = train_data["y"]

    xt_val = val_data["X_temporal"]
    xs_val = val_data["X_static"]
    y_val = val_data["y"]

    # Compute class weights
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    total_samples = len(y_train)
    w_pos = float(total_samples) / (2.0 * max(1, n_pos))
    w_neg = float(total_samples) / (2.0 * max(1, n_neg))
    logger.info(
        "Train class distribution: Pos=%d (%.1f%%, weight=%.3f), Neg=%d (%.1f%%, weight=%.3f)",
        n_pos,
        (n_pos / total_samples) * 100,
        w_pos,
        n_neg,
        (n_neg / total_samples) * 100,
        w_neg,
    )

    # Prepare PyTorch datasets
    train_dataset = TensorDataset(
        torch.as_tensor(xt_train, dtype=torch.float32),
        torch.as_tensor(xs_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32).unsqueeze(1),
    )
    val_dataset = TensorDataset(
        torch.as_tensor(xt_val, dtype=torch.float32),
        torch.as_tensor(xs_val, dtype=torch.float32),
        torch.as_tensor(y_val, dtype=torch.float32).unsqueeze(1),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    model = GlucoseLSTM(
        temporal_dim=xt_train.shape[2],
        static_dim=xs_train.shape[1],
        hidden_dim=128,
        num_lstm_layers=2,
        dropout=0.3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0

    w_pos_tensor = torch.tensor(w_pos, device=device, dtype=torch.float32)
    w_neg_tensor = torch.tensor(w_neg, device=device, dtype=torch.float32)

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0

        for b_xt, b_xs, b_y in train_loader:
            b_xt, b_xs, b_y = b_xt.to(device), b_xs.to(device), b_y.to(device)
            optimizer.zero_grad()

            pred = model(b_xt, b_xs)
            weights = torch.where(b_y == 1.0, w_pos_tensor, w_neg_tensor)
            loss = nn.functional.binary_cross_entropy(pred, b_y, weight=weights)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

        scheduler.step()
        avg_train_loss = train_loss_sum / max(1, train_batches)

        # Validation pass
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_val_preds = []
        all_val_y = []

        with torch.no_grad():
            for b_xt, b_xs, b_y in val_loader:
                b_xt, b_xs, b_y = b_xt.to(device), b_xs.to(device), b_y.to(device)
                pred = model(b_xt, b_xs)
                weights = torch.where(b_y == 1.0, w_pos_tensor, w_neg_tensor)
                loss = nn.functional.binary_cross_entropy(pred, b_y, weight=weights)

                val_loss_sum += loss.item()
                val_batches += 1
                all_val_preds.append(pred.cpu().numpy())
                all_val_y.append(b_y.cpu().numpy())

        avg_val_loss = val_loss_sum / max(1, val_batches)
        val_preds_flat = np.vstack(all_val_preds).flatten()
        val_y_flat = np.vstack(all_val_y).flatten()
        val_auc = roc_auc_score(val_y_flat, val_preds_flat)
        val_pr_auc = average_precision_score(val_y_flat, val_preds_flat)

        logger.info(
            "Epoch [%2d/%2d] | Train Loss: %.4f | Val Loss: %.4f | Val ROC-AUC: %.4f | Val PR-AUC: %.4f | LR: %.2e",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_loss,
            val_auc,
            val_pr_auc,
            scheduler.get_last_lr()[0],
        )

        # Early stopping check
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", epoch)
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    logger.info("LSTM Training completed in %.2f seconds.", time.time() - t_start)
    return model


# -----------------------------------------------------------------------------
# XGBoost Training Loop
# -----------------------------------------------------------------------------


def train_xgboost(
    train_data: Dict[str, Any],
    val_data: Dict[str, Any],
    metadata: Dict[str, Any],
    max_train_samples: Optional[int] = 80000,
) -> GlucoseXGBoost:
    """
    Train GlucoseXGBoost model with rolling summary statistics,
    class weighting, and histogram tree learning.
    """
    logger.info("=" * 70)
    logger.info("TRAINING GLUCOSE-XGBOOST CLASSIFIER")
    logger.info("=" * 70)

    n_train = len(train_data["y"])
    if max_train_samples and max_train_samples < n_train:
        indices = np.random.RandomState(42).choice(n_train, size=max_train_samples, replace=False)
        xt_train = train_data["X_temporal"][indices]
        xs_train = train_data["X_static"][indices]
        y_train = train_data["y"][indices]
        logger.info("Subsampled XGBoost training set to %d samples.", max_train_samples)
    else:
        xt_train = train_data["X_temporal"]
        xs_train = train_data["X_static"]
        y_train = train_data["y"]

    xt_val = val_data["X_temporal"]
    xs_val = val_data["X_static"]
    y_val = val_data["y"]

    dyn_names = metadata.get("dynamic_features", None)
    stat_names = metadata.get("static_features", None)

    logger.info("Extracting rolling summary statistics for XGBoost...")
    t0 = time.time()
    X_train_flat, feat_names = GlucoseXGBoost.extract_tabular_features(
        xt_train, xs_train, dynamic_names=dyn_names, static_names=stat_names
    )
    X_val_flat, _ = GlucoseXGBoost.extract_tabular_features(
        xt_val, xs_val, dynamic_names=dyn_names, static_names=stat_names
    )
    logger.info(
        "Extracted %d tabular features across %d train and %d val samples in %.2fs.",
        X_train_flat.shape[1],
        X_train_flat.shape[0],
        X_val_flat.shape[0],
        time.time() - t0,
    )

    xgb_model = GlucoseXGBoost(
        n_estimators=180,
        max_depth=6,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight="auto",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        feature_names=feat_names,
    )

    t_start = time.time()
    xgb_model.train(
        X_train_flat,
        y_train,
        eval_set=[(X_val_flat, y_val)],
        early_stopping_rounds=15,
        verbose=False,
    )
    logger.info("XGBoost training completed in %.2f seconds.", time.time() - t_start)
    return xgb_model


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }
    return metrics


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def run_training_pipeline(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    lstm_epochs: int = 10,
    max_train_samples: int = 80000,
) -> GlucoTwinEnsemble:
    """Execute complete model training, evaluation, and artifact saving pipeline."""
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "processed"
    if output_dir is None:
        output_dir = data_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    train_data, val_data, test_data, metadata = load_dataset(data_dir)

    # 1. Train LSTM Branch
    lstm_model = train_lstm(
        train_data=train_data,
        val_data=val_data,
        device=device,
        epochs=lstm_epochs,
        batch_size=512,
        learning_rate=1e-3,
        patience=4,
        max_train_samples=max_train_samples,
    )

    # 2. Train XGBoost Branch
    xgb_model = train_xgboost(
        train_data=train_data,
        val_data=val_data,
        metadata=metadata,
        max_train_samples=max_train_samples,
    )

    # 3. Create GlucoTwin Ensemble
    logger.info("=" * 70)
    logger.info("CONSTRUCTING GLUCOTWIN MULTIMODAL ENSEMBLE (LSTM 0.6 + XGBOOST 0.4)")
    logger.info("=" * 70)
    ensemble = GlucoTwinEnsemble(
        lstm_model=lstm_model,
        xgb_model=xgb_model,
        lstm_weight=0.6,
        xgb_weight=0.4,
        device=device,
    )

    # 4. Comprehensive Evaluation on Test Set
    logger.info("Evaluating models on held-out test cohort (%d sequences)...", len(test_data["y"]))
    xt_test = test_data["X_temporal"]
    xs_test = test_data["X_static"]
    y_test = test_data["y"]

    # Pre-extract test features for XGBoost
    dyn_names = metadata.get("dynamic_features", None)
    stat_names = metadata.get("static_features", None)
    X_test_flat, _ = GlucoseXGBoost.extract_tabular_features(
        xt_test, xs_test, dynamic_names=dyn_names, static_names=stat_names
    )

    # Model predictions on test set
    lstm_probs = lstm_model.predict_proba(xt_test, xs_test, device=device, batch_size=1024)
    xgb_probs = xgb_model.predict_proba(X_test_flat, return_2d=False)
    ensemble_probs = ensemble.predict_proba(xt_test, xs_test, x_flat=X_test_flat)

    lstm_metrics = evaluate_model(y_test, lstm_probs)
    xgb_metrics = evaluate_model(y_test, xgb_probs)
    ensemble_metrics = evaluate_model(y_test, ensemble_probs)

    all_metrics = {
        "evaluation_cohort_size": len(y_test),
        "test_spike_ratio_pct": round(float(np.mean(y_test) * 100), 2),
        "lstm_model": lstm_metrics,
        "xgboost_model": xgb_metrics,
        "glucotwin_ensemble": ensemble_metrics,
    }

    # Print evaluation leaderboard
    print("\n" + "=" * 78)
    print("           GLUCOTWIN 2-HOUR GLUCOSE SPIKE PREDICTION BENCHMARK            ")
    print("=" * 78)
    print(f"{'Metric':<22} | {'LSTM (0.6)':<14} | {'XGBoost (0.4)':<14} | {'Ensemble':<14}")
    print("-" * 78)
    print(f"{'ROC-AUC':<22} | {lstm_metrics['roc_auc']:<14.4f} | {xgb_metrics['roc_auc']:<14.4f} | {ensemble_metrics['roc_auc']:<14.4f}")
    print(f"{'PR-AUC (Avg Precision)':<22} | {lstm_metrics['pr_auc']:<14.4f} | {xgb_metrics['pr_auc']:<14.4f} | {ensemble_metrics['pr_auc']:<14.4f}")
    print(f"{'Accuracy':<22} | {lstm_metrics['accuracy']:<14.4f} | {xgb_metrics['accuracy']:<14.4f} | {ensemble_metrics['accuracy']:<14.4f}")
    print(f"{'Sensitivity / Recall':<22} | {lstm_metrics['recall_sensitivity']:<14.4f} | {xgb_metrics['recall_sensitivity']:<14.4f} | {ensemble_metrics['recall_sensitivity']:<14.4f}")
    print(f"{'Specificity':<22} | {lstm_metrics['specificity']:<14.4f} | {xgb_metrics['specificity']:<14.4f} | {ensemble_metrics['specificity']:<14.4f}")
    print(f"{'F1-Score':<22} | {lstm_metrics['f1_score']:<14.4f} | {xgb_metrics['f1_score']:<14.4f} | {ensemble_metrics['f1_score']:<14.4f}")
    print(f"{'Brier Score':<22} | {lstm_metrics['brier_score']:<14.4f} | {xgb_metrics['brier_score']:<14.4f} | {ensemble_metrics['brier_score']:<14.4f}")
    print("=" * 78)

    # 5. Save Artifacts
    logger.info("Saving trained models and evaluation artifacts to %s...", output_dir)
    ensemble.save(output_dir)

    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    fi_df = xgb_model.get_feature_importance(top_n=30)
    fi_path = output_dir / "feature_importance.json"
    fi_df.to_json(fi_path, orient="records", indent=2)

    logger.info("Saved evaluation metrics to %s", metrics_path)
    logger.info("Saved feature importance to %s", fi_path)
    logger.info("GlucoTwin training and evaluation pipeline completed successfully!")

    return ensemble


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GlucoTwin LSTM and XGBoost models.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of LSTM training epochs (default: 10)."
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=80000,
        help="Maximum training samples to use for fast hackathon demo training (default: 80000).",
    )
    args = parser.parse_args()

    run_training_pipeline(lstm_epochs=args.epochs, max_train_samples=args.max_samples)
