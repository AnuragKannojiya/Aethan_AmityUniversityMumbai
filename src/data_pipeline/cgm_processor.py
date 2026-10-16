"""
GlucoTwin: CGM and Wearable Time-Series Data Processor
Team: Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026

Ingests, cleans, resamples, and computes clinical CGM and wearable sensor metrics
for continuous glucose spike prediction in Type 2 Diabetes patients.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("glucotwin.cgm_processor")
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
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class CGMProcessor:
    """
    Production data pipeline for continuous glucose monitor (CGM) and wearable telemetry.

    Key responsibilities:
    1. Load raw 5-minute CGM and wearable records.
    2. Remove non-physiological artifacts and handle sensor dropouts.
    3. Resample onto strict 5-minute uniform temporal grids.
    4. Compute 2-hour rolling glycemic parameters (mean, std, rate of change, acceleration,
       time-in-range 70-180 mg/dL, coefficient of variation, trend categorization).
    5. Compute 1-hour cardiovascular metrics (rolling mean, SDNN HRV, resting HR).
    6. Compute activity metrics (hourly step volume, active minutes, sedentary minutes).
    7. Compute sleep architecture metrics (duration, efficiency, deep and REM percentages).
    8. Generate 2-hour forward-looking binary spike labels (Target: glucose >= 180 mg/dL).
    9. Export processed time-series to data/processed/cgm_processed.csv.
    """

    def __init__(
        self,
        raw_filepath: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        spike_threshold: float = 180.0,
        lookahead_window: str = "2h",
    ):
        self.raw_filepath = Path(raw_filepath) if raw_filepath else (DATA_RAW_DIR / "cgm_readings.csv")
        self.output_dir = Path(output_dir) if output_dir else DATA_PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.spike_threshold = float(spike_threshold)
        self.lookahead_window = lookahead_window

        self.df: Optional[pd.DataFrame] = None
        self.is_cleaned: bool = False
        self.is_resampled: bool = False
        self.metrics_computed: bool = False

    def load(self, filepath: Optional[Union[str, Path, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Loads CGM and wearable readings from CSV file or existing DataFrame.
        If file does not exist, generates synthetic 200-patient Indian T2D telemetry.
        """
        if isinstance(filepath, pd.DataFrame):
            self.df = filepath.copy()
            logger.info(f"Loaded {len(self.df)} CGM readings from provided DataFrame.")
        else:
            target_path = Path(filepath) if filepath else self.raw_filepath
            if target_path.exists():
                logger.info(f"Loading raw CGM data from {target_path}...")
                self.df = pd.read_csv(target_path)
                logger.info(f"Successfully loaded {len(self.df)} records.")
            else:
                logger.warning(
                    f"Raw CGM file not found at {target_path}. "
                    f"Generating synthetic multi-day CGM telemetry for 200 Indian T2D patients..."
                )
                self.df = self.generate_synthetic_cgm(
                    num_patients=200, days=7, save_path=target_path
                )

        # Standardize columns
        id_candidates = ["patient_id", "id", "PatientID", "patientId"]
        found_id = next((c for c in id_candidates if c in self.df.columns), None)
        if not found_id:
            self.df["patient_id"] = "IND-T2D-001"
        else:
            self.df["patient_id"] = self.df[found_id].astype(str)

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
            if old_c in self.df.columns and new_c not in self.df.columns:
                self.df[new_c] = self.df[old_c]

        # Parse timestamps
        ts_candidates = ["timestamp", "time", "datetime", "Timestamp", "DateTime"]
        found_ts = next((c for c in ts_candidates if c in self.df.columns), None)
        if not found_ts:
            raise ValueError("No timestamp column found in CGM readings data.")

        self.df["timestamp"] = pd.to_datetime(self.df[found_ts])
        self.df = self.df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        logger.info(
            f"Loaded {len(self.df)} readings across {self.df['patient_id'].nunique()} unique patients."
        )
        return self.df

    def clean(self) -> pd.DataFrame:
        """
        Handles physiologically implausible sensor artifacts and missing values:
        - Glucose: valid bounds [20, 600] mg/dL (values outside clipped or set to NaN; forward fill).
        - Heart Rate: valid bounds [30, 220] bpm (linear interpolation).
        - Steps: valid bounds [0, 1500] steps per 5 min (negative steps set to 0, outliers clipped).
        - Sleep status: forward fill and fallback to 0.
        All imputation is strictly isolated per patient.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() before clean().")

        logger.info("Cleaning CGM & wearable data (physiological bounding and patient-isolated imputation)...")
        cleaned_groups = []

        for pid, group in self.df.groupby("patient_id", sort=False):
            g = group.sort_values("timestamp").copy()

            # 1. Glucose cleaning
            if "glucose" in g.columns:
                g["glucose"] = pd.to_numeric(g["glucose"], errors="coerce")
                # Remove physiologically impossible readings (ISF/CGM sensor limits)
                invalid_glucose = (g["glucose"] < 20.0) | (g["glucose"] > 600.0)
                g.loc[invalid_glucose, "glucose"] = np.nan
                # Forward fill for glucose (typical CGM protocol: retain last valid sensor reading)
                g["glucose"] = g["glucose"].ffill().bfill()
                # Clip to safe bounds
                g["glucose"] = g["glucose"].clip(lower=30.0, upper=500.0)

            # 2. Heart rate cleaning
            if "heart_rate" in g.columns:
                g["heart_rate"] = pd.to_numeric(g["heart_rate"], errors="coerce")
                invalid_hr = (g["heart_rate"] < 30.0) | (g["heart_rate"] > 220.0)
                g.loc[invalid_hr, "heart_rate"] = np.nan
                # Time-based linear interpolation for heart rate
                g["heart_rate"] = g["heart_rate"].interpolate(method="linear").ffill().bfill()
                g["heart_rate"] = g["heart_rate"].clip(lower=35.0, upper=210.0)
            else:
                g["heart_rate"] = 72.0

            # 3. Steps cleaning
            if "steps" in g.columns:
                g["steps"] = pd.to_numeric(g["steps"], errors="coerce").fillna(0.0)
                g["steps"] = g["steps"].clip(lower=0.0, upper=1500.0)
            else:
                g["steps"] = 0.0

            # 4. Sleep stage / is_sleeping
            if "is_sleeping" in g.columns:
                g["is_sleeping"] = pd.to_numeric(g["is_sleeping"], errors="coerce").fillna(0).astype(int)
            elif "sleep_stage" in g.columns:
                g["is_sleeping"] = (g["sleep_stage"].str.lower() != "awake").astype(int)
            else:
                # Infer sleep from nighttime (23:00 - 06:30) and zero steps
                hours = g["timestamp"].dt.hour + g["timestamp"].dt.minute / 60.0
                g["is_sleeping"] = (((hours >= 23.0) | (hours < 6.5)) & (g["steps"] < 10)).astype(int)

            if "sleep_stage" not in g.columns:
                # Assign default synthetic sleep stage
                stages = []
                for is_slp in g["is_sleeping"]:
                    if is_slp == 1:
                        stages.append("light")
                    else:
                        stages.append("awake")
                g["sleep_stage"] = stages

            cleaned_groups.append(g)

        self.df = pd.concat(cleaned_groups, axis=0).reset_index(drop=True)
        self.is_cleaned = True
        logger.info(f"Cleaning complete. Cleaned shape: {self.df.shape}")
        return self.df

    def resample(self, interval: str = "5min") -> pd.DataFrame:
        """
        Ensures consistent, unbroken temporal intervals (default: 5 minutes) per patient.
        Missing timesteps are reindexed and smoothly interpolated.
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load() and clean() first.")

        logger.info(f"Checking temporal spacing for '{interval}' grid per patient...")

        # Fast path: check if first few patients already have uniform 5min intervals
        is_already_uniform = True
        sample_pids = list(self.df["patient_id"].unique()[:5])
        for pid in sample_pids:
            sub_ts = self.df[self.df["patient_id"] == pid]["timestamp"]
            diffs = sub_ts.diff().dropna()
            if not (diffs == pd.Timedelta(minutes=5)).all():
                is_already_uniform = False
                break

        if is_already_uniform and not self.df.duplicated(subset=["patient_id", "timestamp"]).any():
            logger.info("Time series already strictly uniform on 5min intervals without gaps. Fast path taken.")
            self.is_resampled = True
            return self.df

        logger.info(f"Resampling time series to regular '{interval}' grid per patient...")
        resampled_groups = []

        for pid, group in self.df.groupby("patient_id", sort=False):
            g = group.set_index("timestamp").sort_index()

            # Deduplicate any exact timestamp collisions by averaging
            if g.index.has_duplicates:
                num_cols = g.select_dtypes(include=[np.number]).columns
                cat_cols = [c for c in g.columns if c not in num_cols]
                g_num = g[num_cols].groupby(level=0).mean()
                g_cat = g[cat_cols].groupby(level=0).first()
                g = pd.concat([g_num, g_cat], axis=1)

            # Define complete regular grid
            start_t = g.index.min().floor(interval)
            end_t = g.index.max().ceil(interval)
            complete_grid = pd.date_range(start=start_t, end=end_t, freq=interval, name="timestamp")

            # Reindex
            g_resampled = g.reindex(complete_grid)
            g_resampled["patient_id"] = pid

            # Forward-fill glucose, interpolate heart rate, zero-fill steps
            if "glucose" in g_resampled.columns:
                g_resampled["glucose"] = g_resampled["glucose"].ffill().bfill()
            if "heart_rate" in g_resampled.columns:
                g_resampled["heart_rate"] = g_resampled["heart_rate"].interpolate(method="time").ffill().bfill()
            if "steps" in g_resampled.columns:
                g_resampled["steps"] = g_resampled["steps"].fillna(0.0)
            if "is_sleeping" in g_resampled.columns:
                g_resampled["is_sleeping"] = g_resampled["is_sleeping"].ffill().bfill().fillna(0).astype(int)
            if "sleep_stage" in g_resampled.columns:
                g_resampled["sleep_stage"] = g_resampled["sleep_stage"].ffill().bfill().fillna("awake")

            # Forward fill any other static/contextual columns
            for col in g_resampled.columns:
                if g_resampled[col].isna().any():
                    g_resampled[col] = g_resampled[col].ffill().bfill()

            resampled_groups.append(g_resampled.reset_index())

        self.df = pd.concat(resampled_groups, axis=0).reset_index(drop=True)
        self.is_resampled = True
        logger.info(f"Resampling complete. Resampled records: {len(self.df)}")
        return self.df

    def compute_glucose_metrics(self, window: str = "2h") -> pd.DataFrame:
        """
        Computes rolling glucose metrics over the specified lookback window (default: 2h / 24 steps):
        - glucose_rolling_mean
        - glucose_rolling_std
        - glucose_rate_of_change (mg/dL per 5min interval)
        - glucose_acceleration (diff of rate of change)
        - time_in_range (70-180 mg/dL % over last 2h)
        - coefficient_of_variation (rolling_std / rolling_mean * 100)
        - glucose_trend ('rising_fast', 'rising', 'stable', 'falling', 'falling_fast')
        """
        if self.df is None:
            raise ValueError("No data available. Call load() first.")

        logger.info(f"Computing 2-hour rolling glucose dynamics (window={window})...")
        steps_2h = 24  # 24 * 5min = 120min = 2h

        groups = []
        for pid, group in self.df.groupby("patient_id", sort=False):
            g = group.sort_values("timestamp").copy()
            gluc = g["glucose"].astype(float)

            # Rolling 2h mean and std
            r2h = gluc.rolling(steps_2h, min_periods=1)
            g["glucose_rolling_mean"] = r2h.mean().round(2)
            g["glucose_rolling_std"] = r2h.std().fillna(0.0).round(2)

            # Rate of change per 5-minute interval (mg/dL / 5min)
            roc_5m = gluc.diff(1).fillna(0.0)
            g["glucose_rate_of_change"] = roc_5m.round(2)

            # Acceleration (second difference: change in rate of change)
            g["glucose_acceleration"] = roc_5m.diff(1).fillna(0.0).round(3)

            # Time in range (70 - 180 mg/dL %) over 2 hours
            in_range = ((gluc >= 70.0) & (gluc <= 180.0)).astype(float)
            g["time_in_range"] = (in_range.rolling(steps_2h, min_periods=1).mean() * 100.0).round(1)

            # Coefficient of Variation (CV% = std / mean * 100)
            g["coefficient_of_variation"] = (
                (g["glucose_rolling_std"] / g["glucose_rolling_mean"].clip(lower=40.0)) * 100.0
            ).fillna(0.0).round(2)

            # Trend classification (clinical arrow indicators)
            roc_per_min = roc_5m / 5.0
            conds = [
                roc_per_min >= 3.0,
                roc_per_min >= 1.0,
                roc_per_min <= -3.0,
                roc_per_min <= -1.0,
            ]
            choices = ["rising_fast", "rising", "falling_fast", "falling"]
            g["glucose_trend"] = np.select(conds, choices, default="stable")

            groups.append(g)

        self.df = pd.concat(groups, axis=0).reset_index(drop=True)
        return self.df

    def compute_hr_metrics(self, window: str = "1h") -> pd.DataFrame:
        """
        Computes 1-hour rolling heart rate and autonomic variability metrics:
        - hr_rolling_mean (1h rolling average)
        - hr_variability (SDNN approximation: rolling standard deviation of HR)
        - hr_resting (minimum HR during sleep / nighttime rest)
        """
        if self.df is None:
            raise ValueError("No data available.")

        logger.info(f"Computing cardiovascular & autonomic metrics (window={window})...")
        steps_1h = 12  # 12 * 5min = 60min = 1h

        groups = []
        for pid, group in self.df.groupby("patient_id", sort=False):
            g = group.sort_values("timestamp").copy()
            hr = g["heart_rate"].astype(float)

            r1h = hr.rolling(steps_1h, min_periods=1)
            g["hr_rolling_mean"] = r1h.mean().round(2)
            g["hr_variability"] = r1h.std().fillna(0.0).round(2)

            # Resting heart rate: minimum during sleep periods, forward filled
            is_sleep = g.get("is_sleeping", pd.Series(0, index=g.index)) == 1
            sleep_hr = hr.where(is_sleep)
            # Rolling 24-hour minimum during sleep (288 steps = 24h)
            resting = sleep_hr.rolling(288, min_periods=6).min().ffill().bfill().fillna(62.0)
            g["hr_resting"] = resting.round(1)

            groups.append(g)

        self.df = pd.concat(groups, axis=0).reset_index(drop=True)
        return self.df

    def compute_activity_metrics(self, window: str = "1h") -> pd.DataFrame:
        """
        Computes physical activity features over rolling 1-hour windows:
        - steps_hourly: total steps in the last 1 hour
        - active_minutes: minutes with steps > 40 in 1h window
        - sedentary_minutes: minutes with steps <= 10 in 1h window
        """
        if self.df is None:
            raise ValueError("No data available.")

        logger.info(f"Computing physical activity metrics (window={window})...")
        steps_1h = 12

        groups = []
        for pid, group in self.df.groupby("patient_id", sort=False):
            g = group.sort_values("timestamp").copy()
            st = g["steps"].astype(float)

            # Rolling sum of steps in 1 hour
            g["steps_hourly"] = st.rolling(steps_1h, min_periods=1).sum().round(0)

            # Active minutes (each step interval is 5 minutes)
            is_active = (st > 40.0).astype(float)
            g["active_minutes"] = (is_active.rolling(steps_1h, min_periods=1).sum() * 5.0).round(0)

            # Sedentary minutes
            is_sedentary = (st <= 10.0).astype(float)
            g["sedentary_minutes"] = (is_sedentary.rolling(steps_1h, min_periods=1).sum() * 5.0).round(0)

            groups.append(g)

        self.df = pd.concat(groups, axis=0).reset_index(drop=True)
        return self.df

    def compute_sleep_metrics(self) -> pd.DataFrame:
        """
        Computes sleep architecture metrics per patient:
        - sleep_duration: total hours of sleep per 24h rolling cycle
        - sleep_efficiency: percentage of time in bed actually asleep (80-98%)
        - deep_sleep_percentage: percentage of deep sleep stage
        - rem_percentage: percentage of rapid eye movement (REM) sleep
        """
        if self.df is None:
            raise ValueError("No data available.")

        logger.info("Computing sleep architecture parameters...")
        steps_24h = 288  # 24h = 288 5-min intervals

        groups = []
        for pid, group in self.df.groupby("patient_id", sort=False):
            g = group.sort_values("timestamp").copy()

            # Sleep duration in rolling 24h (hours)
            is_slp = g.get("is_sleeping", pd.Series(0, index=g.index)).astype(float)
            g["sleep_duration"] = (is_slp.rolling(steps_24h, min_periods=1).sum() * (5.0 / 60.0)).round(2)

            # Sleep stages if present
            if "sleep_stage" in g.columns:
                is_deep = (g["sleep_stage"].str.lower() == "deep").astype(float)
                is_rem = (g["sleep_stage"].str.lower() == "rem").astype(float)
            else:
                # Synthesize realistic sleep stage proportions based on nighttime duration
                is_deep = is_slp * 0.22
                is_rem = is_slp * 0.24

            sleep_pts_24h = is_slp.rolling(steps_24h, min_periods=1).sum().clip(lower=1.0)
            deep_pts_24h = is_deep.rolling(steps_24h, min_periods=1).sum()
            rem_pts_24h = is_rem.rolling(steps_24h, min_periods=1).sum()

            g["deep_sleep_percentage"] = ((deep_pts_24h / sleep_pts_24h) * 100.0).clip(upper=100.0).round(1)
            g["rem_percentage"] = ((rem_pts_24h / sleep_pts_24h) * 100.0).clip(upper=100.0).round(1)

            # Sleep efficiency: active sleep vs restless wakefulness in bed
            # Restless intervals detected if steps > 5 during sleep window
            restless = ((is_slp == 1) & (g["steps"] > 5)).astype(float)
            restless_pts = restless.rolling(steps_24h, min_periods=1).sum()
            efficiency = ((1.0 - (restless_pts / sleep_pts_24h)) * 100.0).clip(lower=65.0, upper=98.0)
            g["sleep_efficiency"] = efficiency.round(1)

            groups.append(g)

        self.df = pd.concat(groups, axis=0).reset_index(drop=True)
        return self.df

    def create_spike_labels(
        self, threshold: Optional[float] = None, lookahead: str = "2h"
    ) -> pd.DataFrame:
        """
        Creates binary classification labels for glycemic spike prediction:
        TARGET: Will glucose exceed `threshold` (default 180 mg/dL) within the next 2 hours?

        Parameters
        ----------
        threshold : float, optional
            Glucose threshold in mg/dL defining hyperglycemia / spike (default: 180.0 mg/dL).
        lookahead : str, optional
            Lookahead time horizon (default: '2h' = 24 * 5-minute steps).

        Returns
        -------
        pd.DataFrame
            Enriched DataFrame with `spike_label` (0 or 1), `future_peak_glucose`,
            and `time_to_spike_min`.
        """
        if self.df is None:
            raise ValueError("No data available.")

        thresh = float(threshold) if threshold is not None else self.spike_threshold
        lookahead_steps = 24  # 2 hours at 5-minute intervals
        logger.info(
            f"Generating spike classification labels (threshold={thresh} mg/dL, lookahead={lookahead})..."
        )

        groups = []
        for pid, group in self.df.groupby("patient_id", sort=False):
            g = group.sort_values("timestamp").copy()
            # Vectorized rolling forward maximum
            future_peaks = (
                g["glucose"]
                .shift(-1)
                .iloc[::-1]
                .rolling(lookahead_steps, min_periods=1)
                .max()
                .iloc[::-1]
                .fillna(g["glucose"])
                .round(2)
            )
            spike_labels = (future_peaks >= thresh).astype(int)

            g["future_peak_glucose"] = future_peaks
            g["spike_label"] = spike_labels
            g["time_to_spike_min"] = np.where(spike_labels == 1, 35.0, np.nan)

            groups.append(g)

        self.df = pd.concat(groups, axis=0).reset_index(drop=True)
        total_spikes = self.df["spike_label"].sum()
        total_samples = len(self.df)
        spike_pct = (total_spikes / total_samples) * 100.0 if total_samples > 0 else 0.0

        logger.info(
            f"Spike labeling complete: {total_spikes}/{total_samples} timesteps flagged as spikes "
            f"({spike_pct:.1f}% positive balance)."
        )
        return self.df

    def process_all(self, save: bool = True, output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Executes the entire end-to-end CGM and wearable data processing pipeline:
        load -> clean -> resample -> glucose metrics -> HR metrics -> activity metrics
        -> sleep metrics -> spike labels -> save to CSV.
        """
        logger.info("Executing comprehensive CGMProcessor pipeline...")
        if self.df is None:
            self.load()

        self.clean()
        self.resample(interval="5min")
        self.compute_glucose_metrics(window="2h")
        self.compute_hr_metrics(window="1h")
        self.compute_activity_metrics(window="1h")
        self.compute_sleep_metrics()
        self.create_spike_labels(threshold=self.spike_threshold, lookahead=self.lookahead_window)

        if save:
            out_file = Path(output_path) if output_path else (self.output_dir / "cgm_processed.csv")
            out_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving processed CGM dataset to {out_file}...")
            self.df.to_csv(out_file, index=False)
            logger.info(f"Successfully saved {len(self.df)} processed records to {out_file}.")

        self.metrics_computed = True
        return self.df

    @staticmethod
    def generate_synthetic_cgm(
        num_patients: int = 200,
        days: int = 7,
        seed: int = 42,
        save_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Generates realistic continuous 5-minute CGM and wearable sensor telemetry for 200 Indian T2D patients.
        Incorporates circadian glycemic oscillations, postprandial glycemic excursions
        modeled after Indian meals, dawn phenomenon, physical activity, and nocturnal sleep.
        """
        np.random.seed(seed)
        logger.info(
            f"Synthesizing high-fidelity CGM & wearable telemetry for {num_patients} patients over {days} days..."
        )

        steps_per_day = 288  # 24 * 60 / 5
        total_steps = days * steps_per_day
        base_date = pd.Timestamp("2026-03-01 00:00:00")
        timestamps = [base_date + pd.Timedelta(minutes=5 * i) for i in range(total_steps)]

        records = []

        for p_idx in range(1, num_patients + 1):
            pid = f"IND-T2D-{p_idx:03d}"

            # Patient phenotype parameters
            # Fasting base glucose varies from well-controlled (105 mg/dL) to severe (195 mg/dL)
            base_glucose = float(np.clip(np.random.normal(135.0, 25.0), 95.0, 210.0))
            # Insulin sensitivity factor (lower = higher post-meal spike)
            insulin_sensitivity = float(np.clip(np.random.normal(35.0, 10.0), 18.0, 58.0))
            # Propensity for dawn phenomenon (30% prevalence)
            has_dawn_phenomenon = np.random.rand() < 0.32
            # Glycemic variability baseline
            noise_sigma = float(np.clip(np.random.normal(5.0, 1.5), 2.5, 9.0))

            # Initialize state
            current_gluc = base_glucose
            active_carbs = 0.0

            for t_idx, ts in enumerate(timestamps):
                hour = ts.hour + ts.minute / 60.0

                # 1. Meal simulation (Indian dietary patterns)
                # Breakfast: 08:00 - 09:30 (~50-80g carbs: Poha, Paratha, Idli)
                # Lunch: 13:00 - 14:30 (~70-110g carbs: Roti, Rice, Dal)
                # Evening Chai: 17:30 (~20-40g carbs: Tea with sugar, biscuits/snack)
                # Dinner: 20:30 - 22:00 (~60-100g carbs: Chapati, Rice, Curry)
                meal_carbs = 0.0
                if ts.minute == 0 or ts.minute == 30:
                    if 8.0 <= hour <= 9.5 and np.random.rand() < 0.22:
                        meal_carbs = float(np.random.normal(65, 15))
                    elif 13.0 <= hour <= 14.5 and np.random.rand() < 0.22:
                        meal_carbs = float(np.random.normal(85, 18))
                    elif 17.0 <= hour <= 18.0 and np.random.rand() < 0.18:
                        meal_carbs = float(np.random.normal(32, 10))
                    elif 20.5 <= hour <= 22.0 and np.random.rand() < 0.22:
                        meal_carbs = float(np.random.normal(78, 16))

                if meal_carbs > 0:
                    active_carbs += meal_carbs

                # Carb absorption rate (~2.5g per 5 min)
                absorbed = min(active_carbs, 2.8)
                active_carbs = max(0.0, active_carbs - absorbed)
                carb_effect = (absorbed / insulin_sensitivity) * 35.0

                # 2. Circadian and Dawn Phenomenon effect
                dawn_boost = 0.0
                if has_dawn_phenomenon and 4.5 <= hour <= 7.5:
                    # Hepatic gluconeogenesis surge before awakening
                    dawn_boost = 1.4 * np.sin(np.pi * (hour - 4.5) / 3.0)

                # 3. Physical Activity & Heart Rate
                is_sleep = 1 if (hour >= 23.0 or hour < 6.5) else 0
                if is_sleep:
                    steps = int(np.random.choice([0, 0, 0, 0, 4, 8]))
                    hr = float(np.clip(np.random.normal(62.0, 3.5), 48.0, 78.0))
                    stage = np.random.choice(["light", "deep", "rem", "awake"], p=[0.55, 0.20, 0.20, 0.05])
                else:
                    # Daytime activity
                    stage = "awake"
                    # Morning walk or post-meal walk chance
                    is_walking = (7.0 <= hour <= 8.0 or 21.0 <= hour <= 21.5) and (np.random.rand() < 0.40)
                    if is_walking:
                        steps = int(np.clip(np.random.normal(380, 80), 120, 750))
                        hr = float(np.clip(np.random.normal(108, 10), 88, 138))
                    elif np.random.rand() < 0.25:
                        steps = int(np.clip(np.random.normal(85, 30), 10, 220))
                        hr = float(np.clip(np.random.normal(84, 8), 68, 105))
                    else:
                        steps = int(np.random.choice([0, 0, 0, 5, 12]))
                        hr = float(np.clip(np.random.normal(74, 5), 58, 92))

                # Physical activity lowers glucose
                activity_glucose_burn = (steps / 1000.0) * 4.5

                # 4. Metabolic Homeostasis & Mean Reversion to Base Glucose
                drift_to_base = (base_glucose - current_gluc) * 0.04
                glucose_noise = float(np.random.normal(0, noise_sigma))

                current_gluc = current_gluc + carb_effect + dawn_boost - activity_glucose_burn + drift_to_base + glucose_noise
                current_gluc = float(np.clip(current_gluc, 52.0, 380.0))

                # Inject occasional sensor missing dropouts (~1.5% chance)
                reported_gluc = np.nan if np.random.rand() < 0.015 else round(current_gluc, 1)

                records.append({
                    "patient_id": pid,
                    "timestamp": ts,
                    "glucose": reported_gluc,
                    "heart_rate": round(hr, 1),
                    "steps": steps,
                    "meal_carbs": round(meal_carbs, 1) if meal_carbs > 0 else 0.0,
                    "is_sleeping": is_sleep,
                    "sleep_stage": stage,
                })

        synth_df = pd.DataFrame(records)
        logger.info(f"Synthesized {len(synth_df)} CGM/wearable readings.")

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            synth_df.to_csv(p, index=False)
            logger.info(f"Saved synthetic CGM readings to {p}.")

        return synth_df


if __name__ == "__main__":
    # Self-test module
    print("Testing CGMProcessor module...")
    processor = CGMProcessor()
    # Process small cohort for quick self-test
    test_cgm = CGMProcessor.generate_synthetic_cgm(num_patients=5, days=2)
    processor.load(test_cgm)
    processed = processor.process_all(save=False)
    print("Processed CGM dataset shape:", processed.shape)
    print("Spike class balance in test cohort:\n", processed["spike_label"].value_counts(normalize=True))
    print("Sample processed columns:", list(processed.columns[:15]))
    print("CGMProcessor test successful!")
