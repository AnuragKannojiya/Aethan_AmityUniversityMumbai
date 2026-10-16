"""GlucoTwin: Multimodal Data Fusion & Sequence Pipeline"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_pipeline.cgm_processor import CGMProcessor
from src.feature_engineering.feature_store import FeatureStore
from src.feature_engineering.static_features import (
    extract_static_features,
    generate_synthetic_ehr_cohort,
)
from src.feature_engineering.temporal_features import extract_temporal_features

logger = logging.getLogger("glucotwin.data_fusion")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class DataFusion:
    """Multimodal Data Fusion Engine for GlucoTwin."""

    def __init__(
        self,
        ehr_path: Optional[Union[str, Path]] = None,
        cgm_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        seq_length: int = 24,
        prediction_horizon: int = 24,
    ):
        self.ehr_path = Path(ehr_path) if ehr_path else (DATA_RAW_DIR / "ehr_patients.csv")
        self.cgm_path = Path(cgm_path) if cgm_path else (DATA_PROCESSED_DIR / "cgm_processed.csv")
        self.output_dir = Path(output_dir) if output_dir else DATA_PROCESSED_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seq_length = int(seq_length)  # 24 * 5min = 2 hours lookback
        self.prediction_horizon = int(prediction_horizon)  # 2 hours forward prediction

        self.ehr_df: Optional[pd.DataFrame] = None
        self.cgm_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None

        self.feature_store = FeatureStore(store_dir=self.output_dir)
        self.dynamic_features: List[str] = []
        self.static_features: List[str] = []
        self.target_col: str = "spike_label"

        self.splits: Dict[str, Dict[str, Any]] = {}
        self.fusion_summary: Dict[str, Any] = {}

    def load_ehr_features(
        self, path: Optional[Union[str, Path, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Loads EHR records and computes static clinical features."""
        if isinstance(path, pd.DataFrame):
            raw_ehr = path.copy()
            logger.info(f"Loaded EHR records from provided DataFrame (n={len(raw_ehr)}).")
        else:
            target_file = Path(path) if path else self.ehr_path
            if target_file.exists():
                logger.info(f"Loading raw EHR records from {target_file}...")
                raw_ehr = pd.read_csv(target_file)
            else:
                logger.warning(
                    f"EHR file not found at {target_file}. "
                    f"Generating synthetic 200 Indian T2D patient cohort..."
                )
                raw_ehr = generate_synthetic_ehr_cohort(num_patients=200)
                target_file.parent.mkdir(parents=True, exist_ok=True)
                raw_ehr.to_csv(target_file, index=False)
                logger.info(f"Saved synthetic EHR cohort to {target_file}.")

        # Extract rich clinical static features
        self.ehr_df = extract_static_features(raw_ehr)
        logger.info(f"Loaded static EHR features for {len(self.ehr_df)} patients.")
        return self.ehr_df

    def load_cgm_features(
        self, path: Optional[Union[str, Path, pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """Loads processed CGM and wearable time-series data."""
        if isinstance(path, pd.DataFrame):
            self.cgm_df = path.copy()
            logger.info(f"Loaded CGM records from provided DataFrame (n={len(self.cgm_df)}).")
        else:
            target_file = Path(path) if path else self.cgm_path
            if target_file.exists():
                logger.info(f"Loading processed CGM data from {target_file}...")
                self.cgm_df = pd.read_csv(target_file)
                self.cgm_df["timestamp"] = pd.to_datetime(self.cgm_df["timestamp"])
            else:
                logger.warning(
                    f"Processed CGM file not found at {target_file}. Running CGMProcessor pipeline..."
                )
                processor = CGMProcessor(output_dir=self.output_dir)
                self.cgm_df = processor.process_all(save=True, output_path=target_file)

        # Ensure temporal features are enriched
        self.cgm_df = extract_temporal_features(self.cgm_df)

        # Standardize patient_id type
        self.cgm_df["patient_id"] = self.cgm_df["patient_id"].astype(str)
        logger.info(
            f"Loaded CGM & wearable features: {len(self.cgm_df)} readings across "
            f"{self.cgm_df['patient_id'].nunique()} patients."
        )
        return self.cgm_df

    def merge(self, strategy: str = "broadcast") -> pd.DataFrame:
        """Merges static EHR features with dynamic CGM time-series."""
        if self.ehr_df is None:
            self.load_ehr_features()
        if self.cgm_df is None:
            self.load_cgm_features()

        logger.info(f"Merging EHR and CGM data using '{strategy}' strategy...")

        # Ensure patient_id is consistent string
        self.ehr_df["patient_id"] = self.ehr_df["patient_id"].astype(str)
        self.cgm_df["patient_id"] = self.cgm_df["patient_id"].astype(str)

        # Determine non-conflicting static columns to merge
        # Exclude patient_id from static columns list to prevent duplication
        static_cols_to_add = [
            c for c in self.ehr_df.columns if c != "patient_id" and c not in self.cgm_df.columns
        ]

        ehr_subset = self.ehr_df[["patient_id"] + static_cols_to_add].drop_duplicates(subset=["patient_id"])

        self.merged_df = pd.merge(
            self.cgm_df,
            ehr_subset,
            on="patient_id",
            how="inner",
        ).sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

        logger.info(
            f"Merged dataset shape: {self.merged_df.shape} "
            f"({self.merged_df['patient_id'].nunique()} shared patients)."
        )

        # Update feature lists from FeatureStore schema
        all_cols = set(self.merged_df.columns)
        self.dynamic_features = [
            c for c in self.feature_store.dynamic_feature_cols if c in all_cols
        ]
        self.static_features = [
            c for c in self.feature_store.static_feature_cols if c in all_cols
        ]

        logger.info(
            f"Active feature schema: {len(self.dynamic_features)} dynamic features, "
            f"{len(self.static_features)} static features."
        )
        return self.merged_df

    def create_sequences(
        self,
        seq_length: Optional[int] = None,
        prediction_horizon: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generates sliding window sequence tensors for multi-input deep learning models:"""
        if self.merged_df is None:
            self.merge()

        L = int(seq_length) if seq_length is not None else self.seq_length
        H = int(prediction_horizon) if prediction_horizon is not None else self.prediction_horizon
        actual_stride = int(stride) if stride is not None else (6 if len(self.merged_df) > 50000 else 1)

        logger.info(
            f"Generating sliding sequences (lookback L={L} steps [2h], horizon H={H} steps [2h], stride={actual_stride})..."
        )

        X_temporal_list: List[np.ndarray] = []
        X_static_list: List[np.ndarray] = []
        y_list: List[int] = []
        patient_id_list: List[str] = []
        timestamp_list: List[str] = []

        for pid, group in self.merged_df.groupby("patient_id", sort=False):
            g = group.sort_values("timestamp").reset_index(drop=True)
            n_rows = len(g)
            if n_rows < L:
                continue

            dyn_mat = g[self.dynamic_features].fillna(0.0).values.astype(np.float32)
            stat_vec = g[self.static_features].iloc[0].fillna(0.0).values.astype(np.float32)
            labels = g[self.target_col].values.astype(int)
            ts_series = g["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").values

            # Sliding window from L-1 to n_rows - 1 with actual_stride
            for end_idx in range(L - 1, n_rows, actual_stride):
                start_idx = end_idx - L + 1
                temporal_window = dyn_mat[start_idx : end_idx + 1]
                target_label = labels[end_idx]
                target_ts = ts_series[end_idx]

                X_temporal_list.append(temporal_window)
                X_static_list.append(stat_vec)
                y_list.append(target_label)
                patient_id_list.append(pid)
                timestamp_list.append(target_ts)

        X_temporal = np.array(X_temporal_list, dtype=np.float32)
        X_static = np.array(X_static_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)
        pids = np.array(patient_id_list)
        ts_arr = np.array(timestamp_list)

        logger.info(
            f"Created sequence matrices: X_temporal shape={X_temporal.shape}, "
            f"X_static shape={X_static.shape}, y shape={y.shape}"
        )
        return X_temporal, X_static, y, pids, ts_arr

    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stride: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Splits data by PATIENT (Group-based partitioning) to avoid data leakage."""
        if self.merged_df is None:
            self.merge()

        unique_patients = np.array(sorted(self.merged_df["patient_id"].unique()))
        num_patients = len(unique_patients)
        actual_stride = int(stride) if stride is not None else (6 if len(self.merged_df) > 50000 else 1)
        logger.info(
            f"Partitioning {num_patients} patients into Train/Val/Test cohorts (stride={actual_stride})..."
        )

        # Split patients into Train+Val vs Test
        train_val_patients, test_patients = train_test_split(
            unique_patients, test_size=test_size, random_state=random_state
        )

        # Split Train+Val into Train vs Val
        val_fraction_adjusted = val_size / (1.0 - test_size)
        train_patients, val_patients = train_test_split(
            train_val_patients, test_size=val_fraction_adjusted, random_state=random_state
        )

        logger.info(
            f"Patient cohort split: Train={len(train_patients)}, "
            f"Val={len(val_patients)}, Test={len(test_patients)}"
        )

        # Fit FeatureStore scalers exclusively on training patients
        self.feature_store.fit(
            static_df=self.ehr_df,
            temporal_df=self.cgm_df,
            train_patient_ids=list(train_patients),
        )

        # Scale dynamic and static features in merged_df
        scaled_merged_df = self.merged_df.copy()
        scaled_dyn = self.feature_store.transform_temporal(scaled_merged_df, scale=True)
        scaled_stat = self.feature_store.transform_static(scaled_merged_df, scale=True)

        for i, col in enumerate(self.dynamic_features):
            scaled_merged_df[col] = scaled_dyn[:, i]
        for j, col in enumerate(self.static_features):
            scaled_merged_df[col] = scaled_stat[:, j]

        # Generate sequences from scaled merged dataset per split
        splits_dict: Dict[str, Dict[str, Any]] = {}
        cohorts = {
            "train": train_patients,
            "val": val_patients,
            "test": test_patients,
        }

        L = self.seq_length

        for split_name, cohort_pids in cohorts.items():
            pid_set = set(cohort_pids)
            sub_df = scaled_merged_df[scaled_merged_df["patient_id"].isin(pid_set)].reset_index(drop=True)

            X_temp_list = []
            X_stat_list = []
            y_sub_list = []
            pid_sub_list = []
            ts_sub_list = []

            for pid, g in sub_df.groupby("patient_id", sort=False):
                g = g.sort_values("timestamp").reset_index(drop=True)
                n = len(g)
                if n < L:
                    continue

                dyn_vals = g[self.dynamic_features].values.astype(np.float32)
                stat_vals = g[self.static_features].iloc[0].values.astype(np.float32)
                y_vals = g[self.target_col].values.astype(int)
                ts_vals = g["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").values

                for end_i in range(L - 1, n, actual_stride):
                    st_i = end_i - L + 1
                    X_temp_list.append(dyn_vals[st_i : end_i + 1])
                    X_stat_list.append(stat_vals)
                    y_sub_list.append(y_vals[end_i])
                    pid_sub_list.append(pid)
                    ts_sub_list.append(ts_vals[end_i])

            X_temp_arr = np.array(X_temp_list, dtype=np.float32)
            X_stat_arr = np.array(X_stat_list, dtype=np.float32)
            y_arr = np.array(y_sub_list, dtype=np.int64)

            spike_count = int(np.sum(y_arr))
            total_seqs = len(y_arr)
            spike_pct = (spike_count / total_seqs * 100.0) if total_seqs > 0 else 0.0

            splits_dict[split_name] = {
                "X_temporal": X_temp_arr,
                "X_static": X_stat_arr,
                "y": y_arr,
                "patient_ids": np.array(pid_sub_list),
                "timestamps": np.array(ts_sub_list),
                "cohort_patient_ids": np.array(cohort_pids),
                "num_sequences": total_seqs,
                "spike_count": spike_count,
                "spike_percentage": round(spike_pct, 2),
            }

            logger.info(
                f"[{split_name.upper()}] Patients: {len(cohort_pids)}, Sequences: {total_seqs}, "
                f"Spikes: {spike_count} ({spike_pct:.1f}%)"
            )

        self.splits = splits_dict

        self.fusion_summary = {
            "total_patients": int(num_patients),
            "train_patients": int(len(train_patients)),
            "val_patients": int(len(val_patients)),
            "test_patients": int(len(test_patients)),
            "dynamic_features_count": len(self.dynamic_features),
            "static_features_count": len(self.static_features),
            "sequence_length": self.seq_length,
            "prediction_horizon": self.prediction_horizon,
            "train_sequences": splits_dict["train"]["num_sequences"],
            "val_sequences": splits_dict["val"]["num_sequences"],
            "test_sequences": splits_dict["test"]["num_sequences"],
            "train_spike_pct": splits_dict["train"]["spike_percentage"],
            "val_spike_pct": splits_dict["val"]["spike_percentage"],
            "test_spike_pct": splits_dict["test"]["spike_percentage"],
        }

        return self.splits

    def save_splits(self, output_dir: Optional[Union[str, Path]] = None) -> Dict[str, Path]:
        """Saves partitioned datasets as high-performance compressed .npz archives"""
        out_dir = Path(output_dir) if output_dir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if not self.splits:
            raise RuntimeError("No splits available. Call split_data() first.")

        saved_paths: Dict[str, Path] = {}

        for split_name in ["train", "val", "test"]:
            split_data = self.splits[split_name]
            npz_path = out_dir / f"{split_name}_data.npz"

            np.savez_compressed(
                npz_path,
                X_temporal=split_data["X_temporal"],
                X_static=split_data["X_static"],
                y=split_data["y"],
                patient_ids=split_data["patient_ids"],
                timestamps=split_data["timestamps"],
                cohort_patient_ids=split_data["cohort_patient_ids"],
            )
            saved_paths[split_name] = npz_path
            logger.info(f"Saved {split_name} split to {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.2f} MB).")

        # Save FeatureStore scalers and metadata
        self.feature_store.save(out_dir / "feature_store.joblib")

        # Save feature schema and fusion summary
        schema_path = out_dir / "feature_metadata.json"
        meta_payload = {
            "summary": self.fusion_summary,
            "dynamic_features": self.dynamic_features,
            "static_features": self.static_features,
            "target": self.target_col,
        }
        with open(schema_path, "w") as f:
            json.dump(meta_payload, f, indent=2)
        saved_paths["metadata"] = schema_path

        logger.info(f"Saved feature metadata and schema to {schema_path}.")
        return saved_paths

    def get_feature_names(self) -> Dict[str, List[str]]:
        """Returns ordered feature names for model explanation and SHAP analysis."""
        return {
            "dynamic_features": list(self.dynamic_features),
            "static_features": list(self.static_features),
            "target": [self.target_col],
        }

    def fuse_all(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
        stride: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Executes the complete multimodal data fusion pipeline:"""
        logger.info("Starting complete GlucoTwin multimodal fusion pipeline...")
        self.load_ehr_features()
        self.load_cgm_features()
        self.merge(strategy="broadcast")
        self.split_data(
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            stride=stride,
        )
        saved_paths = self.save_splits()

        logger.info("================ GLUCOTWIN FUSION PIPELINE COMPLETE ================")
        for k, v in self.fusion_summary.items():
            logger.info(f"  {k}: {v}")
        logger.info("====================================================================")

        return {
            "summary": self.fusion_summary,
            "saved_paths": {k: str(v) for k, v in saved_paths.items()},
            "feature_names": self.get_feature_names(),
        }


if __name__ == "__main__":
    print("Testing DataFusion module...")
    # Initialize with synthetic test data for quick validation
    fusion = DataFusion()
    results = fusion.fuse_all(test_size=0.2, val_size=0.1)
    print("\nData Fusion Results Summary:")
    print(json.dumps(results["summary"], indent=2))
    print("\nDataFusion test successful!")
