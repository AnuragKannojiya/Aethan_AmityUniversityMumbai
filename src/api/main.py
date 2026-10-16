from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.routes import get_global_service, router as api_router
from src.api.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = get_global_service()
    if not service.is_initialized:
        service.initialize()
    app.state.service = service
    yield


app = FastAPI(
    title="GlucoTwin API",
    description="Digital Twin API for Type 2 Diabetes Glucose Spike Prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
async def root() -> Dict[str, Any]:
    service = getattr(app.state, "service", None) or get_global_service()
    unique_count = len({p.get("patient_id") for p in service.patients_dict.values()})
    return {
        "title": "GlucoTwin API",
        "version": "1.0.0",
        "status": "online",
        "patients_enrolled": unique_count,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def system_health() -> HealthResponse:
    service = getattr(app.state, "service", None) or get_global_service()
    unique_count = len({p.get("patient_id") for p in service.patients_dict.values()})
    return HealthResponse(
        status="ok",
        version="1.0.0",
        service="GlucoTwin API",
        patients_loaded=unique_count,
        demo_mode=service.demo_mode,
        timestamp=datetime.now(timezone.utc),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
