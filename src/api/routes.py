from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

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

_global_service: Optional[GlucoTwinService] = None


def get_global_service() -> GlucoTwinService:
    global _global_service
    if _global_service is None:
        _global_service = GlucoTwinService()
        if not _global_service.is_initialized:
            _global_service.initialize()
    return _global_service


def get_service(request: Request) -> GlucoTwinService:
    if hasattr(request.app.state, "service") and request.app.state.service is not None:
        return request.app.state.service
    return get_global_service()


class FlexibleInterventionRequest(BaseModel):
    patient_id: Optional[str] = None
    intervention_type: Optional[str] = None
    intervention: Optional[str] = None
    intensity: Optional[str] = "moderate"
    walk_minutes: Optional[int] = None
    meal_carbs_g: Optional[int] = None
    medication_taken: Optional[bool] = None


router = APIRouter(prefix="/api", tags=["GlucoTwin"])


@router.get("/patients", response_model=List[PatientSummary])
async def list_patients(
    risk_level: Optional[str] = Query(None),
    city: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    service: GlucoTwinService = Depends(get_service),
) -> List[PatientSummary]:
    return service.get_all_patients(
        risk_level=risk_level, city=city, search=search, limit=limit, offset=offset
    )


@router.get("/patients/{patient_id}", response_model=PatientDetail)
async def get_patient_profile(
    patient_id: str, service: GlucoTwinService = Depends(get_service)
) -> PatientDetail:
    patient = service.get_patient(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return patient


@router.get("/patients/{patient_id}/readings", response_model=List[GlucoseReading])
async def get_cgm_readings(
    patient_id: str,
    hours: int = Query(24, ge=1, le=720),
    service: GlucoTwinService = Depends(get_service),
) -> List[GlucoseReading]:
    if service.get_patient(patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return service.get_recent_readings(patient_id, hours=hours)


@router.get("/patients/{patient_id}/predict", response_model=PredictionResponse)
@router.get("/patients/{patient_id}/prediction", response_model=PredictionResponse, include_in_schema=False)
async def predict_glucose(
    patient_id: str, service: GlucoTwinService = Depends(get_service)
) -> PredictionResponse:
    prediction = service.predict_glucose_trajectory(patient_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return prediction


@router.post("/patients/{patient_id}/simulate", response_model=InterventionResponse)
async def simulate_intervention_endpoint(
    patient_id: str,
    payload: FlexibleInterventionRequest,
    service: GlucoTwinService = Depends(get_service),
) -> InterventionResponse:
    if service.get_patient(patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    itype = payload.intervention_type or payload.intervention or "exercise"
    req = InterventionRequest(
        patient_id=patient_id,
        intervention_type=itype,
        intensity=payload.intensity or "moderate",
        walk_minutes=payload.walk_minutes,
        meal_carbs_g=payload.meal_carbs_g,
        medication_taken=payload.medication_taken,
    )
    return service.simulate_intervention(patient_id, req)


@router.get("/patients/{patient_id}/alerts", response_model=List[AlertResponse])
async def get_patient_alerts_endpoint(
    patient_id: str, service: GlucoTwinService = Depends(get_service)
) -> List[AlertResponse]:
    if service.get_patient(patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return service.get_patient_alerts(patient_id)


@router.get("/patients/{patient_id}/dashboard", response_model=DashboardData)
async def get_dashboard_endpoint(
    patient_id: str, service: GlucoTwinService = Depends(get_service)
) -> DashboardData:
    data = service.get_dashboard_data(patient_id)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return data


@router.get("/patients/{patient_id}/glucose-stats", response_model=GlucoseStats)
async def get_glucose_stats_endpoint(
    patient_id: str,
    hours: int = Query(48, ge=1, le=720),
    service: GlucoTwinService = Depends(get_service),
) -> GlucoseStats:
    stats = service.get_glucose_statistics(patient_id, hours=hours)
    if stats is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return stats


@router.get("/stats/overview", response_model=SystemOverviewStats)
async def get_system_overview_endpoint(
    service: GlucoTwinService = Depends(get_service),
) -> SystemOverviewStats:
    return service.get_system_overview()


@router.get("/stats/evaluation")
async def get_evaluation_metrics_endpoint(
    service: GlucoTwinService = Depends(get_service),
):
    return service.get_evaluation_metrics()


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: GlucoTwinService = Depends(get_service),
) -> HealthResponse:
    unique_count = len({p.get("patient_id") for p in service.patients_dict.values()})
    return HealthResponse(
        status="ok",
        version="1.0.0",
        service="GlucoTwin API",
        patients_loaded=unique_count,
        demo_mode=service.demo_mode,
        timestamp=datetime.now(timezone.utc),
    )
