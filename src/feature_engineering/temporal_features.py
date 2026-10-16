"""GlucoTwin: Temporal CGM & Wearable Feature Engineering Module"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("glucotwin.temporal_features")
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


def _estimate_meal_events(
    patient_df: pd.DataFrame,
    glucose_col: str = "glucose",
    steps_col: str = "steps",
    window_steps: int = 6,
) -> pd.Series:
    """Estimates post-prandial meal events from physiological glucose excursion patterns"""
    glucose = patient_df[glucose_col]
    steps = patient_df[steps_col] if steps_col in patient_df.columns else pd.Series(0, index=patient_df.index)
    ts = pd.to_datetime(patient_df["timestamp"])

    hours = ts.dt.hour + ts.dt.minute / 60.0
    in_meal_hours = (
        ((hours >= 7.5) & (hours <= 10.5))
        | ((hours >= 12.5) & (hours <= 15.5))
        | ((hours >= 17.0) & (hours <= 18.5))
        | ((hours >= 19.5) & (hours <= 22.75))
    )

    diff_30m = glucose - glucose.shift(window_steps)
    roc_5m = (glucose - glucose.shift(1)) / 5.0
    accel = roc_5m - roc_5m.shift(1)
    low_exercise = steps.rolling(window_steps, min_periods=1).sum() < 600

    # Detection logic
    is_detected = (
        (diff_30m >= 20.0)
        & (roc_5m > 0.8)
        & (accel > 0.0)
        & in_meal_hours
        & low_exercise
    )

    # If explicit meal carbs column exists, include it
    if "meal_carbs" in patient_df.columns:
        explicit_meal = patient_df["meal_carbs"].fillna(0) > 0
        is_detected = is_detected | explicit_meal

    # Suppress consecutive positive detections within 60 minutes
    suppressed = is_detected.copy()
    last_idx = -999
    for i, flag in enumerate(is_detected):
        if flag:
            if i - last_idx < 12:  # within 60 minutes
                suppressed.iloc[i] = False
            else:
                last_idx = i

    return suppressed.astype(int)


def _compute_patient_temporal_features(group: pd.DataFrame) -> pd.DataFrame:
    """Computes all temporal features for a single patient time-series."""
    df = group.sort_values("timestamp").copy()
    ts = pd.to_datetime(df["timestamp"])

    # 1. Time of Day Encoding (Sin / Cos)
    hours = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    df["sin_time_of_day"] = np.sin(2.0 * np.pi * hours / 24.0).round(4)
    df["cos_time_of_day"] = np.cos(2.0 * np.pi * hours / 24.0).round(4)

    # Clinical diurnal period indicators
    df["is_dawn_period"] = ((hours >= 4.0) & (hours <= 7.5)).astype(int)
    df["is_nighttime"] = ((hours >= 23.0) | (hours < 6.0)).astype(int)

    # 2. Day of Week Encoding
    dow = ts.dt.dayofweek
    df["sin_day_of_week"] = np.sin(2.0 * np.pi * dow / 7.0).round(4)
    df["cos_day_of_week"] = np.cos(2.0 * np.pi * dow / 7.0).round(4)
    df["is_weekend"] = (dow >= 5).astype(int)

    # Ensure essential columns exist
    if "glucose" not in df.columns:
        raise ValueError("CGM dataframe must contain 'glucose' column.")
    if "heart_rate" not in df.columns:
        df["heart_rate"] = 75.0
    if "steps" not in df.columns:
        df["steps"] = 0.0

    glucose = df["glucose"].astype(float)
    hr = df["heart_rate"].astype(float)
    steps = df["steps"].astype(float)

    # 3. Multi-scale Glucose Features
    # 5-minute (1 step)
    g_diff_5m = glucose.diff(1)
    g_roc_5m = (g_diff_5m / 5.0).fillna(0.0)
    g_accel_5m = (g_roc_5m.diff(1) / 5.0).fillna(0.0)

    df["glucose_diff_5m"] = g_diff_5m.fillna(0.0).round(2)
    df["glucose_rate_of_change"] = g_roc_5m.round(3)
    df["glucose_acceleration"] = g_accel_5m.round(4)

    # 15-minute (3 steps)
    r15 = glucose.rolling(3, min_periods=1)
    df["glucose_mean_15m"] = r15.mean().round(2)
    df["glucose_std_15m"] = r15.std().fillna(0.0).round(2)
    df["glucose_diff_15m"] = (glucose - glucose.shift(3)).fillna(0.0).round(2)
    df["glucose_min_15m"] = r15.min().round(2)
    df["glucose_max_15m"] = r15.max().round(2)

    # 30-minute (6 steps)
    r30 = glucose.rolling(6, min_periods=1)
    df["glucose_mean_30m"] = r30.mean().round(2)
    df["glucose_std_30m"] = r30.std().fillna(0.0).round(2)
    df["glucose_diff_30m"] = (glucose - glucose.shift(6)).fillna(0.0).round(2)
    df["glucose_min_30m"] = r30.min().round(2)
    df["glucose_max_30m"] = r30.max().round(2)

    # 1-hour (12 steps)
    r1h = glucose.rolling(12, min_periods=1)
    df["glucose_mean_1h"] = r1h.mean().round(2)
    df["glucose_std_1h"] = r1h.std().fillna(0.0).round(2)
    df["glucose_diff_1h"] = (glucose - glucose.shift(12)).fillna(0.0).round(2)
    df["glucose_min_1h"] = r1h.min().round(2)
    df["glucose_max_1h"] = r1h.max().round(2)
    df["glucose_range_1h"] = (df["glucose_max_1h"] - df["glucose_min_1h"]).round(2)

    # 2-hour (24 steps)
    r2h = glucose.rolling(24, min_periods=1)
    df["glucose_rolling_mean"] = r2h.mean().round(2)
    df["glucose_rolling_std"] = r2h.std().fillna(0.0).round(2)
    df["glucose_diff_2h"] = (glucose - glucose.shift(24)).fillna(0.0).round(2)
    df["glucose_min_2h"] = r2h.min().round(2)
    df["glucose_max_2h"] = r2h.max().round(2)
    df["glucose_range_2h"] = (df["glucose_max_2h"] - df["glucose_min_2h"]).round(2)

    # Coefficient of Variation (CV% over 2h)
    df["coefficient_of_variation"] = (
        (df["glucose_rolling_std"] / df["glucose_rolling_mean"].clip(lower=40.0)) * 100.0
    ).fillna(0.0).round(2)

    # Time in Range (70-180 mg/dL % over last 2h)
    in_range = ((glucose >= 70.0) & (glucose <= 180.0)).astype(float)
    df["time_in_range"] = (in_range.rolling(24, min_periods=1).mean() * 100.0).round(1)

    # Time above range (> 180 mg/dL)
    above_range = (glucose > 180.0).astype(float)
    df["time_above_range"] = (above_range.rolling(24, min_periods=1).mean() * 100.0).round(1)

    # 4. Glucose Momentum (Weighted Rate of Change)
    # EMA fast (span=3, ~15m) vs slow (span=12, ~60m)
    ema_fast = glucose.ewm(span=3, adjust=False).mean()
    ema_slow = glucose.ewm(span=12, adjust=False).mean()
    df["glucose_momentum_fast"] = ((ema_fast - ema_fast.shift(1)) / 5.0).fillna(0.0).round(3)
    df["glucose_momentum_slow"] = ((ema_slow - ema_slow.shift(1)) / 5.0).fillna(0.0).round(3)
    df["glucose_momentum_divergence"] = (ema_fast - ema_slow).round(2)

    # 5. Glucose Volatility Index (GVI)
    # Mean Absolute Glucose rate (MAG): sum of |diff| over last 1h / 60 min
    mag_1h = g_diff_5m.abs().rolling(12, min_periods=1).sum() / 60.0
    df["glucose_mag_1h"] = mag_1h.round(3)

    # Custom Volatility Index combining CV, momentum, and MAG
    df["glucose_volatility_index"] = (
        (df["coefficient_of_variation"] / 100.0)
        * (1.0 + df["glucose_momentum_fast"].abs())
        * (1.0 + df["glucose_mag_1h"])
    ).round(4)

    # 6. Post-Prandial Glucose Excursion Magnitude
    # Excursion above rolling baseline nadir
    df["postprandial_excursion_magnitude"] = (
        np.maximum(0.0, glucose - df["glucose_min_2h"])
    ).round(2)
    df["postprandial_excursion_ratio"] = (
        df["postprandial_excursion_magnitude"] / df["glucose_min_2h"].clip(lower=50.0)
    ).round(3)

    # 7. Meal Proximity Features
    meal_event_flag = _estimate_meal_events(df, glucose_col="glucose", steps_col="steps")
    df["is_meal_event"] = meal_event_flag

    # Vectorized computation of minutes elapsed since last meal
    meal_indices = np.where(meal_event_flag.values == 1)[0]
    n_pts = len(df)
    if len(meal_indices) == 0:
        time_since_meal = np.full(n_pts, 360.0, dtype=np.float32)
    else:
        idx_arr = np.arange(n_pts)
        pos = np.searchsorted(meal_indices, idx_arr, side="right") - 1
        valid_mask = pos >= 0
        time_since_meal = np.full(n_pts, 360.0, dtype=np.float32)
        time_since_meal[valid_mask] = np.minimum(360.0, (idx_arr[valid_mask] - meal_indices[pos[valid_mask]]) * 5.0)

    df["estimated_time_since_meal_min"] = time_since_meal
    # Meal absorption exponential decay factor
    df["meal_absorption_decay"] = np.exp(-time_since_meal / 60.0).round(4)
    # Postprandial 2-hour window flag
    df["is_postprandial_window"] = (df["estimated_time_since_meal_min"] <= 120.0).astype(int)

    # 8. Nocturnal Glucose Pattern Features
    is_night = df["is_nighttime"] == 1
    night_glucose = glucose.where(is_night)
    # Forward fill nocturnal average into daytime for metabolic reference
    df["nocturnal_glucose_mean"] = (
        night_glucose.rolling(48, min_periods=6).mean().ffill().bfill().round(2)
    )

    # Dawn phenomenon flag: 04:00 - 07:30, rising glucose (> 1.0 mg/dL/min), sedentary steps
    df["dawn_phenomenon_flag"] = (
        (df["is_dawn_period"] == 1)
        & (df["glucose_rate_of_change"] > 0.6)
        & (steps.rolling(6, min_periods=1).sum() < 40)
    ).astype(int)

    # Nocturnal hypoglycemia alert flag
    df["nocturnal_hypo_flag"] = ((is_night) & (glucose < 70.0)).astype(int)

    # 9. Heart Rate Features and Recovery
    r_hr_1h = hr.rolling(12, min_periods=1)
    df["hr_rolling_mean"] = r_hr_1h.mean().round(2)
    df["hr_variability"] = r_hr_1h.std().fillna(0.0).round(2)

    # HR during sleep / resting (10th percentile or minimum during night)
    sleep_hr = hr.where(df["is_nighttime"] == 1)
    resting_hr = sleep_hr.rolling(72, min_periods=6).min().ffill().bfill().fillna(62.0)
    df["hr_resting"] = resting_hr.round(1)

    # Heart Rate Recovery (HRR) after activity
    active_bout = (steps > 100) | (hr > 100)
    hr_when_active = hr.where(active_bout)
    rolling_peak_hr = hr_when_active.rolling(9, min_periods=1).max().ffill().fillna(hr)
    df["hr_recovery_delta"] = np.maximum(0.0, rolling_peak_hr - hr).round(1)

    # 10. Activity & Step Count Trend
    r_steps_1h = steps.rolling(12, min_periods=1)
    df["steps_hourly"] = r_steps_1h.sum().round(0)
    df["active_minutes"] = ((steps > 40.0).astype(float).rolling(12, min_periods=1).sum() * 5.0).round(0)
    df["sedentary_minutes"] = ((steps <= 10.0).astype(float).rolling(12, min_periods=1).sum() * 5.0).round(0)

    # Step count trend over last 2 hours (12 readings)
    steps_recent_30m = steps.rolling(6, min_periods=1).sum()
    steps_prior_30m = steps.shift(6).rolling(6, min_periods=1).sum().fillna(0.0)
    df["step_count_trend"] = (steps_recent_30m - steps_prior_30m).round(1)

    # Glucose trend categorical and encoded (vectorized)
    conds = [
        df["glucose_rate_of_change"] >= 3.0,
        df["glucose_rate_of_change"] >= 1.0,
        df["glucose_rate_of_change"] <= -3.0,
        df["glucose_rate_of_change"] <= -1.0,
    ]
    df["glucose_trend"] = np.select(
        conds, ["rising_fast", "rising", "falling_fast", "falling"], default="stable"
    )
    df["glucose_trend_code"] = np.select(conds, [2, 1, -2, -1], default=0)

    return df


def extract_temporal_features(cgm_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts and engineers temporal dynamic features from continuous glucose"""
    df = cgm_df.copy()

    # Identify patient identifier
    id_col = None
    for cand in ["patient_id", "id", "PatientID", "patientId"]:
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        df["patient_id"] = "PAT_0001"
        id_col = "patient_id"

    df["patient_id"] = df[id_col].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Standardize column mappings from raw telemetry
    col_renames = {
        "glucose_mg_dl": "glucose",
        "glucose_val": "glucose",
        "heart_rate_bpm": "heart_rate",
        "steps_last_5min": "steps",
        "steps_5min": "steps",
        "hr": "heart_rate",
        "heartrate": "heart_rate",
    }
    for old_c, new_c in col_renames.items():
        if old_c in df.columns and new_c not in df.columns:
            df[new_c] = df[old_c]

    logger.info(
        f"Extracting temporal features across {df['patient_id'].nunique()} patient cohorts..."
    )

    # Process each patient group independently to strictly avoid temporal leakage
    grouped_results = []
    for pid, group in df.groupby("patient_id", sort=False):
        processed_group = _compute_patient_temporal_features(group)
        grouped_results.append(processed_group)

    out_df = pd.concat(grouped_results, axis=0).reset_index(drop=True)
    logger.info(
        f"Completed temporal feature extraction: shape={out_df.shape}, "
        f"features_count={out_df.shape[1]}"
    )
    return out_df


if __name__ == "__main__":
    # Self-test module
    print("Testing temporal_features module...")
    dates = pd.date_range("2026-03-01 00:00:00", periods=288, freq="5min")
    np.random.seed(42)
    dummy_cgm = pd.DataFrame({
        "patient_id": ["IND-T2D-001"] * 288,
        "timestamp": dates,
        "glucose": 130.0 + 35.0 * np.sin(np.linspace(0, 4 * np.pi, 288)) + np.random.normal(0, 5, 288),
        "heart_rate": 72.0 + 15.0 * np.random.rand(288),
        "steps": np.random.poisson(30, 288),
    })
    temporal_df = extract_temporal_features(dummy_cgm)
    print("Extracted temporal features shape:", temporal_df.shape)
    print("Sample temporal feature columns:", list(temporal_df.columns[:20]))
    print("Temporal features extraction test successful!")
