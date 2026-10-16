"""
GlucoTwin Feature Engineering Package
=====================================

Static EHR risk profiling, multi-scale temporal CGM feature extraction,
and central feature store management for Indian Type 2 Diabetes digital twins.
"""

from .static_features import (
    extract_static_features,
    generate_synthetic_ehr_cohort,
    COMORBIDITY_SEVERITY_WEIGHTS,
)
from .temporal_features import (
    extract_temporal_features,
)
from .feature_store import (
    FeatureStore,
)

__all__ = [
    "extract_static_features",
    "generate_synthetic_ehr_cohort",
    "extract_temporal_features",
    "FeatureStore",
    "COMORBIDITY_SEVERITY_WEIGHTS",
]
