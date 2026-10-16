"""
GlucoTwin API Routes
====================
FastAPI route definitions for the GlucoTwin Digital Twin platform for Type 2
Diabetes glucose spike prediction, patient monitoring, and counterfactual simulation.

Author: Team Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

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

logger = logging.getLogger("glucotwin.api.routes")

# Module-level service singleton reference
_global_service: Optional[GlucoTwinService] = None


def get_global_service() -> GlucoTwinService:
    """Return or initialize the global GlucoTwin service instance."""
    global _global_service
    if _global_service is None:
        logger.info("Initializing global GlucoTwinService singleton...")
        _global_service = GlucoTwinService()
        if not _global_service.is_initialized:
            _global_service.initialize()
    return _global_service


def get_service(request: Request) -> GlucoTwinService:
    """
    FastAPI dependency injection provider for GlucoTwinService.
    Uses app.state.service if initialized by application lifespan,
    falling back to module singleton.
    """
    if hasattr(request.app.state, "service") and request.app.state.service is not None:
        return request.app.state.service
    return get_global_service()


class FlexibleInterventionRequest(BaseModel):
    """
    Intervention simulation payload accommodating both strict schema
    and frontend dashboard naming conventions.
    """

    patient_id: Optional[str] = Field(
        None,
        description="Target patient identifier (defaults to URL path parameter)",
        json_schema_extra={"example": "IND-T2D-001"},
    )
    intervention_type: Optional[str] = Field(
        None,
        description="Intervention strategy (exercise, take_medication, eat_meal, drink_water)",
        json_schema_extra={"example": "exercise"},
    )
    intervention: Optional[str] = Field(
        None,
        description="Frontend alias for intervention_type",
        json_schema_extra={"example": "take_medication"},
    )
    intensity: Optional[str] = Field(
        "moderate",
        description="Intensity or dosage specification (light, moderate, vigorous, high)",
        json_schema_extra={"example": "moderate"},
    )


router = APIRouter(prefix="/api", tags=["GlucoTwin Digital Twin"])


# -----------------------------------------------------------------------------
# Patient Endpoints
# -----------------------------------------------------------------------------


@router.get(
    "/patients",
    response_model=List[PatientSummary],
    summary="List all patients with summary",
    description="Retrieve a paginated list of monitored Type 2 Diabetes patients with optional filtering by risk tier, city, or search term.",
)
async def list_patients(
    risk_level: Optional[str] = Query(
        None,
        description="Filter by glycemic risk tier (Low, Moderate, High, Critical)",
    ),
    city: Optional[str] = Query(
        None,
        description="Filter by Indian city of residence (e.g. Mumbai, New Delhi, Bengaluru)",
    ),
    search: Optional[str] = Query(
        None,
        description="Case-insensitive substring search matching name, patient ID, or medication",
    ),
    limit: int = Query(
        50,
        ge=1,
        le=500,
        description="Maximum number of patient summary records to return",
    ),
    offset: int = Query(
        0,
        ge=0,
        description="Pagination offset index",
    ),
    service: GlucoTwinService = Depends(get_service),
) -> List[PatientSummary]:
    """List all monitored T2D patients with summary demographic and glycemic vitals."""
    try:
        return service.get_all_patients(
            risk_level=risk_level,
            city=city,
            search=search,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        logger.error("Failed to query patients: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving patients: {str(exc)}",
        )


@router.get(
    "/patients/{patient_id}",
    response_model=PatientDetail,
    summary="Detailed patient profile",
    description="Fetch comprehensive clinical, laboratory, and digital twin profile data for a specific patient.",
)
async def get_patient_profile(
    patient_id: str,
    service: GlucoTwinService = Depends(get_service),
) -> PatientDetail:
    """Retrieve detailed clinical profile including laboratory biomarkers and vitals."""
    patient = service.get_patient(patient_id)
    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with identifier '{patient_id}' not found.",
        )
    return patient


@router.get(
    "/patients/{patient_id}/readings",
    response_model=List[GlucoseReading],
    summary="Recent CGM readings",
    description="Retrieve continuous glucose monitoring (CGM) telemetry and wearable sensor samples for the past N hours.",
)
async def get_cgm_readings(
    patient_id: str,
    hours: int = Query(
        24,
        ge=1,
        le=720,
        description="Telemetry lookback window in hours (default 24h)",
    ),
    service: GlucoTwinService = Depends(get_service),
) -> List[GlucoseReading]:
    """Fetch continuous telemetry stream for patient."""
    # Verify patient exists
    if service.get_patient(patient_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with identifier '{patient_id}' not found.",
        )
    return service.get_recent_readings(patient_id, hours=hours)


# -----------------------------------------------------------------------------
# Prediction & Digital Twin Forecasting Endpoints
# -----------------------------------------------------------------------------


@router.get(
    "/patients/{patient_id}/predict",
    response_model=PredictionResponse,
    summary="Glucose prediction + risk",
    description="Generate 2-hour forward continuous glucose trajectory, calculate spike probability (>180 mg/dL), and generate explainable clinical risk factors.",
)
async def predict_glucose(
    patient_id: str,
    service: GlucoTwinService = Depends(get_service),
) -> PredictionResponse:
    """Forecast 2-hour glucose trajectory and assess acute spike risk."""
    prediction = service.predict_glucose_trajectory(patient_id)
    if prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with identifier '{patient_id}' not found.",
        )
    return prediction


# Alias for frontend dashboard compatibility (/prediction)
@router.get(
    "/patients/{patient_id}/prediction",
    response_model=PredictionResponse,
    include_in_schema=False,
)
async def predict_glucose_alias(
    patient_id: str,
    service: GlucoTwinService = Depends(get_service),
) -> PredictionResponse:
    """Alias for /patients/{patient_id}/predict endpoint."""
    return await predict_glucose(patient_id=patient_id, service=service)


# -----------------------------------------------------------------------------
# Intervention Simulation ('What-If' Digital Twin)
# -----------------------------------------------------------------------------


@router.post(
    "/patients/{patient_id}/simulate",
    response_model=InterventionResponse,
    summary="Simulate intervention",
    description="Run counterfactual Digital Twin simulation to model prospective impact of exercise, medication, meals, or hydration against baseline trajectory.",
)
async def simulate_intervention_endpoint(
    patient_id: str,
    payload: FlexibleInterventionRequest,
    service: GlucoTwinService = Depends(get_service),
) -> InterventionResponse:
    """Simulate lifestyle or pharmacotherapeutic intervention on the Digital Twin."""
    # Verify patient exists
    if service.get_patient(patient_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with identifier '{patient_id}' not found.",
        )

    # Resolve intervention type from either key
    itype = payload.intervention_type or payload.intervention or "exercise"
    intensity = payload.intensity or "moderate"

    req = InterventionRequest(
        patient_id=patient_id,
        intervention_type=itype,
        intensity=intensity,
    )

    result = service.simulate_intervention(patient_id, req)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute counterfactual simulation trajectory.",
        )
    return result


# -----------------------------------------------------------------------------
# Clinical Alerts & Monitoring
# -----------------------------------------------------------------------------


@router.get(
    "/patients/{patient_id}/alerts",
    response_model=List[AlertResponse],
    summary="Active alerts",
    description="Retrieve real-time clinical alerts, impending spike warnings, and acute hypoglycemia notifications for the patient.",
)
async def get_patient_alerts_endpoint(
    patient_id: str,
    service: GlucoTwinService = Depends(get_service),
) -> List[AlertResponse]:
    """Retrieve active clinical alerts and spike warnings."""
    if service.get_patient(patient_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with identifier '{patient_id}' not found.",
        )
    return service.get_patient_alerts(patient_id)


# -----------------------------------------------------------------------------
# Unified Dashboard & Aggregations
# -----------------------------------------------------------------------------


@router.get(
    "/patients/{patient_id}/dashboard",
    response_model=DashboardData,
    summary="Complete dashboard data in one call",
    description="Single consolidated API call delivering patient summary, 24-hour CGM stream, 2-hour forward forecast, active alerts, and glycemic statistics.",
)
async def get_dashboard_endpoint(
    patient_id: str,
    service: GlucoTwinService = Depends(get_service),
) -> DashboardData:
    """Fetch complete unified dashboard state for patient."""
    data = service.get_dashboard_data(patient_id)
    if data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with identifier '{patient_id}' not found.",
        )
    return data


@router.get(
    "/patients/{patient_id}/glucose-stats",
    response_model=GlucoseStats,
    summary="Glucose statistics",
    description="Compute standard glycemic consensus metrics: Time-in-Range (TIR %), TAR, TBR, Mean Glucose, Standard Deviation, CV %, and GMI.",
)
async def get_glucose_stats_endpoint(
    patient_id: str,
    hours: int = Query(
        48,
        ge=1,
        le=720,
        description="Analysis window in hours (default 48h)",
    ),
    service: GlucoTwinService = Depends(get_service),
) -> GlucoseStats:
    """Calculate International Consensus Time-in-Range (TIR) metrics and variability."""
    stats = service.get_glucose_statistics(patient_id, hours=hours)
    if stats is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with identifier '{patient_id}' not found.",
        )
    return stats


# -----------------------------------------------------------------------------
# Population & System Overview
# -----------------------------------------------------------------------------


@router.get(
    "/stats/overview",
    response_model=SystemOverviewStats,
    summary="System-wide statistics",
    description="Cohort-wide population health analytics across all enrolled Type 2 Diabetes patients, risk distributions, and operational pipeline status.",
)
async def get_system_overview_endpoint(
    service: GlucoTwinService = Depends(get_service),
) -> SystemOverviewStats:
    """Retrieve population health metrics and risk breakdown across all 200 patients."""
    return service.get_system_overview()


# -----------------------------------------------------------------------------
# Health Check Endpoint
# -----------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API Health Status",
    description="Check API service operational status, loaded cohort count, UTC timestamp, and heuristic demo engine mode.",
)
async def health_check(
    service: GlucoTwinService = Depends(get_service),
) -> HealthResponse:
    """Return API health status, loaded records count, and active inference engine."""
    # Deduplicate patients count
    unique_count = len({p.get("patient_id") for p in service.patients_dict.values()})
    return HealthResponse(
        status="ok",
        version="1.0.0",
        service="GlucoTwin API",
        patients_loaded=unique_count,
        demo_mode=service.demo_mode,
        timestamp=datetime.now(timezone.utc),
    )
