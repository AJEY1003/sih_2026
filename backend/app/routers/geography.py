from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.data_store import data_store, paginate, to_records

router = APIRouter()

ALLOWED_SORT_COLUMNS = {
    "State", "Constituency", "Constituency_ID", "District_Code",
    "Spatial_Cluster_ID", "Zone_Fraud_Risk_Score",
}


@router.get("", summary="List constituencies with coordinates and zone fraud risk (for map views)")
def list_geography(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    state: Optional[str] = Query(None, description="Exact match, case-insensitive"),
    min_zone_risk: Optional[float] = None,
    sort_by: str = Query("Zone_Fraud_Risk_Score", description=f"One of: {', '.join(sorted(ALLOWED_SORT_COLUMNS))}"),
    sort_desc: bool = True,
):
    if sort_by not in ALLOWED_SORT_COLUMNS:
        raise HTTPException(400, f"Unknown sort_by column: {sort_by!r}")

    df = data_store.geography
    if state:
        df = df[df["State"].str.casefold() == state.casefold()]
    if min_zone_risk is not None:
        df = df[df["Zone_Fraud_Risk_Score"] >= min_zone_risk]

    df = df.sort_values(sort_by, ascending=not sort_desc, na_position="last")
    total, page = paginate(df, skip, limit)
    return {"total": total, "skip": skip, "limit": limit, "items": to_records(page)}


@router.get("/{state}/{constituency_id}", summary="Single constituency's geographic + zone risk detail")
def get_constituency(state: str, constituency_id: str):
    df = data_store.geography
    match = df[
        (df["State"].str.casefold() == state.casefold())
        & (df["Constituency_ID"].astype(str).str.casefold() == constituency_id.casefold())
    ]
    if match.empty:
        raise HTTPException(404, f"Constituency {constituency_id!r} in state {state!r} not found")
    return to_records(match)[0]
