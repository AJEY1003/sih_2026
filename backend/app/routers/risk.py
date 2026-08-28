"""Live inference over HTTP for the Hybrid Risk Engine
(backend/ml_pipeline/hybrid_risk_engine.py) - the "Next Steps" item from the
project README. Model artifacts are loaded lazily, once, on first request
(see hybrid_risk_engine._load_state)."""
from fastapi import APIRouter, HTTPException

from app.schemas import ProjectRiskRequest
from hybrid_risk_engine import generate_investigation_report, predict_project_risk

router = APIRouter()


@router.post(
    "/predict",
    summary="Score a project's fraud/risk 0-100",
    description=(
        "Pass {'Work_ID': ...} to score a known project as recorded in the "
        "dataset, or a dict of raw fields to score a brand-new project not "
        "yet in the dataset. Fields passed alongside Work_ID override the "
        "stored record, enabling 'what if I change X' scenarios."
    ),
)
def predict_risk(payload: ProjectRiskRequest):
    project_data = payload.model_dump(exclude_none=True)
    if not project_data:
        raise HTTPException(400, "Provide at least a Work_ID or one raw project field.")
    try:
        return predict_project_risk(project_data)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Risk scoring failed: {exc}") from exc


@router.get(
    "/investigate/{work_id:path}",
    summary="Full GIS investigation report for a known project",
    description=(
        "Select project -> risk score & level -> WHY (color-coded factors) "
        "-> nearby similar works -> contractor history -> historical cost "
        "benchmark -> recommended action."
    ),
)
def investigate(work_id: str):
    # Work_ID values contain literal slashes - see the note in works.py.
    try:
        return generate_investigation_report(work_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Investigation report failed: {exc}") from exc
