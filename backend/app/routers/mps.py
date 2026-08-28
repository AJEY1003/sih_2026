from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.data_store import data_store, paginate, to_records

router = APIRouter()

ALLOWED_SORT_COLUMNS = {
    "MP_Name", "MP_Total_Allocation", "MP_Total_Sanctioned", "MP_Total_Disbursed",
    "MP_Total_Projects", "MP_Completed_Projects", "MP_Completion_Rate",
    "MP_Budget_Utilization_Percent", "MP_Performance_Rank",
}


@router.get("", summary="List MPs with search, sort and pagination")
def list_mps(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    search: Optional[str] = Query(None, description="Substring match on MP_Name"),
    sort_by: str = Query("MP_Performance_Rank", description=f"One of: {', '.join(sorted(ALLOWED_SORT_COLUMNS))}"),
    sort_desc: bool = False,
):
    if sort_by not in ALLOWED_SORT_COLUMNS:
        raise HTTPException(400, f"Unknown sort_by column: {sort_by!r}")

    df = data_store.mps
    if search:
        df = df[df["MP_Name"].str.casefold().str.contains(search.casefold(), na=False)]

    df = df.sort_values(sort_by, ascending=not sort_desc, na_position="last")
    total, page = paginate(df, skip, limit)
    return {"total": total, "skip": skip, "limit": limit, "items": to_records(page)}


@router.get("/{mp_name}", summary="MP performance detail plus a summary of their works")
def get_mp(mp_name: str):
    df = data_store.mps
    match = df[df["MP_Name"].str.casefold() == mp_name.casefold()]
    if match.empty:
        raise HTTPException(404, f"MP {mp_name!r} not found")
    detail = to_records(match)[0]

    works = data_store.works
    mp_works = works[works["MP_Name"].str.casefold() == mp_name.casefold()]
    detail["works"] = to_records(
        mp_works[["Work_ID", "Work_Status", "State", "Constituency", "Sanction_Amount", "Disbursed_Amount"]]
    )
    return detail
