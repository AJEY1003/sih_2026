"""
MPLADS Analytics & Monitoring Platform - FastAPI backend.

Wraps the ETL star schema (data/output_schema/*.csv) and the Hybrid Risk
Engine (backend/ml_pipeline/hybrid_risk_engine.py) for live HTTP access -
the "Next Steps" item from the project README.

Run from the `backend/` directory:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Interactive docs once running: http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.data_store import data_store
from app.routers import geography, health, mps, risk, stats, vendors, works

app = FastAPI(
    title="MPLADS Analytics & Monitoring API",
    description=(
        "AI-powered monitoring and analytics platform for MPLADS fund "
        "utilization - fraud/anomaly risk scoring plus browsable access to "
        "works, MPs, vendors and geography data."
    ),
    version="1.0.0",
)

# Wide open for now - a Next.js/dashboard frontend is the next planned
# consumer (see README "Next Steps"); tighten allow_origins once that
# frontend's origin is known.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_data() -> None:
    """Eager-loads the dimension/fact CSVs so the first request isn't slow.
    ML model artifacts (Isolation Forest, XGBoost, SBERT) stay lazy-loaded
    on first use inside hybrid_risk_engine - see its _load_state()."""
    data_store.load()


app.include_router(health.router)
app.include_router(risk.router, prefix="/api/risk", tags=["Risk Engine"])
app.include_router(works.router, prefix="/api/works", tags=["Works"])
app.include_router(mps.router, prefix="/api/mps", tags=["Members of Parliament"])
app.include_router(vendors.router, prefix="/api/vendors", tags=["Vendors"])
app.include_router(geography.router, prefix="/api/geography", tags=["Geography"])
app.include_router(stats.router, prefix="/api/stats", tags=["Dashboard Stats"])


@app.get("/", tags=["Health"], summary="API root")
def root():
    return {
        "service": "MPLADS Analytics & Monitoring API",
        "docs": "/docs",
        "health": "/health",
    }
