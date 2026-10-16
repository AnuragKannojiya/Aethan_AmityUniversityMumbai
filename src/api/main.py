"""
GlucoTwin FastAPI Application Server
====================================
FastAPI application entrypoint for GlucoTwin — a Digital Twin platform for
Type 2 Diabetes glucose spike prediction, wearable telemetry fusion, and
counterfactual lifestyle intervention simulations.

Author: Team Aethan, Amity University Mumbai
Challenge: Digital Twin Challenge 2026
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.routes import get_global_service, router as api_router
from src.api.schemas import HealthResponse
from src.api.service import GlucoTwinService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("glucotwin.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Bootstraps the GlucoTwinService on startup by loading raw or processed
    EHR and CGM data and checking model artifacts.
    """
    logger.info("Initializing GlucoTwin Digital Twin backend server...")
    start_time = datetime.now(timezone.utc)

    # Initialize core digital twin service
    service = get_global_service()
    if not service.is_initialized:
        logger.info("Service not initialized yet; running bootstrap...")
        service.initialize()

    # Store reference on app state for dependency injection
    app.state.service = service
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()

    unique_patients = len({p.get("patient_id") for p in service.patients_dict.values()})
    logger.info(
        "GlucoTwin Service ready in %.2fs. Cohort: %d patients. Engine: %s",
        elapsed,
        unique_patients,
        service.model_status,
    )

    yield

    logger.info("Shutting down GlucoTwin Digital Twin server...")


# Initialize FastAPI Application
app = FastAPI(
    title="GlucoTwin API - Digital Twin for Type 2 Diabetes",
    description=(
        "Production-grade Digital Twin REST API fusing static EHR records "
        "and continuous CGM wearable telemetry to forecast glucose spikes "
        "2 hours in advance and simulate counterfactual lifestyle interventions. "
        "Built by Team Aethan, Amity University Mumbai for Digital Twin Challenge 2026."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Configure CORS Middleware (allow all origins, methods, and headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount primary API router (includes /api prefix)
app.include_router(api_router)


# Root endpoint with rich discovery metadata
@app.get(
    "/",
    tags=["System"],
    summary="Root API Discovery",
    description="Returns service description, status, API version, and route directory.",
)
async def root() -> Dict[str, Any]:
    """Root metadata and route index."""
    service: GlucoTwinService = getattr(app.state, "service", None) or get_global_service()
    unique_count = len({p.get("patient_id") for p in service.patients_dict.values()}) if service else 0
    return {
        "title": "GlucoTwin API - Digital Twin for Type 2 Diabetes",
        "service": "GlucoTwin Digital Twin Platform",
        "version": "1.0.0",
        "status": "online",
        "team": "Aethan, Amity University Mumbai",
        "challenge": "Digital Twin Challenge 2026",
        "engine_mode": service.model_status if service else "Initializing",
        "patients_enrolled": unique_count,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "health": "/api/health",
            "patients": "/api/patients",
            "patient_detail": "/api/patients/{patient_id}",
            "cgm_readings": "/api/patients/{patient_id}/readings?hours=24",
            "prediction": "/api/patients/{patient_id}/predict",
            "simulate": "/api/patients/{patient_id}/simulate",
            "alerts": "/api/patients/{patient_id}/alerts",
            "dashboard": "/api/patients/{patient_id}/dashboard",
            "glucose_stats": "/api/patients/{patient_id}/glucose-stats",
            "overview_stats": "/api/stats/overview",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Top-level health check at /health in addition to /api/health
@app.get(
    "/health",
    tags=["System"],
    response_model=HealthResponse,
    summary="System Health Check",
    description="Convenience health check endpoint returning server diagnostic status.",
)
async def system_health() -> HealthResponse:
    """Return system operational status and cohort count."""
    service: GlucoTwinService = getattr(app.state, "service", None) or get_global_service()
    unique_count = len({p.get("patient_id") for p in service.patients_dict.values()}) if service else 0
    return HealthResponse(
        status="ok",
        version="1.0.0",
        service="GlucoTwin API",
        patients_loaded=unique_count,
        demo_mode=service.demo_mode if service else True,
        timestamp=datetime.now(timezone.utc),
    )


# Uvicorn entrypoint for running via `python -m src.api.main`
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
