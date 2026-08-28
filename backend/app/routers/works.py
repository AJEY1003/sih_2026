from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.data_store import data_store, paginate, to_records

router = APIRouter()

ALLOWED_SORT_COLUMNS = {
    "Work_ID", "House", "MP_Name", "Vendor_Name", "State", "Constituency",
    "Work_Status", "Recommended_Date", "Sanction_Date", "Completion_Date",
    "Days_to_Sanction", "Days_to_Complete", "Days_Pending",
    "Recommended_Amount", "Sanction_Amount", "Disbursed_Amount",
    "Cost_Overrun_Amount", "Cost_Overrun_Ratio", "Cost_per_Unit",
}


@router.get("", summary="List works with filters, search, sort and pagination")
def list_works(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    state: Optional[str] = Query(None, description="Exact match, case-insensitive"),
    constituency: Optional[str] = Query(None, description="Exact match, case-insensitive"),
    mp_name: Optional[str] = Query(None, description="Exact match, case-insensitive"),
    vendor_name: Optional[str] = Query(None, description="Exact match, case-insensitive"),
    work_status: Optional[str] = Query(None, description="Exact match, case-insensitive"),
    calamity_flag: Optional[bool] = None,
    search: Optional[str] = Query(None, description="Substring match on Work_ID or Work_Description"),
    min_sanction_amount: Optional[float] = None,
    max_sanction_amount: Optional[float] = None,
    sort_by: str = Query("Work_ID", description=f"One of: {', '.join(sorted(ALLOWED_SORT_COLUMNS))}"),
    sort_desc: bool = False,
):
    if sort_by not in ALLOWED_SORT_COLUMNS:
        raise HTTPException(400, f"Unknown sort_by column: {sort_by!r}")

    df = data_store.works

    if state:
        df = df[df["State"].str.casefold() == state.casefold()]
    if constituency:
        df = df[df["Constituency"].str.casefold() == constituency.casefold()]
    if mp_name:
        df = df[df["MP_Name"].str.casefold() == mp_name.casefold()]
    if vendor_name:
        df = df[df["Vendor_Name"].str.casefold() == vendor_name.casefold()]
    if work_status:
        df = df[df["Work_Status"].str.casefold() == work_status.casefold()]
    if calamity_flag is not None:
        df = df[df["Calamity_Flag"] == calamity_flag]
    if search:
        needle = search.casefold()
        df = df[
            df["Work_ID"].str.casefold().str.contains(needle, na=False)
            | df["Work_Description"].str.casefold().str.contains(needle, na=False)
        ]
    if min_sanction_amount is not None:
        df = df[df["Sanction_Amount"] >= min_sanction_amount]
    if max_sanction_amount is not None:
        df = df[df["Sanction_Amount"] <= max_sanction_amount]

    df = df.sort_values(sort_by, ascending=not sort_desc, na_position="last")
    total, page = paginate(df, skip, limit)
    return {"total": total, "skip": skip, "limit": limit, "items": to_records(page)}


@router.get(
    "/{work_id:path}",
    summary="Full work detail merged with vendor, geography and compliance data",
)
def get_work(work_id: str):
    # Work_ID values contain literal slashes (e.g. "WS/MP005/2024-2025/145074"),
    # so the plain {work_id} converter would only ever capture the first
    # segment - :path lets it match the rest of the URL as-is.
    df = data_store.master
    match = df[df["Work_ID"] == work_id]
    if match.empty:
        raise HTTPException(404, f"Work_ID {work_id!r} not found")
    return to_records(match)[0]
