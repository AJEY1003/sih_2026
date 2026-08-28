from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.data_store import data_store, paginate, to_records

router = APIRouter()

ALLOWED_SORT_COLUMNS = {
    "Vendor_Name", "Vendor_Total_Projects_Count", "Vendor_Avg_Timeline_Days",
    "Vendor_Avg_Cost_Overrun_Ratio", "Vendor_Quality_Rating",
    "Vendor_Monopoly_Score", "Vendor_Risk_Score",
}


@router.get("", summary="List vendors with search, sort and pagination")
def list_vendors(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    search: Optional[str] = Query(None, description="Substring match on Vendor_Name"),
    min_risk_score: Optional[float] = None,
    sort_by: str = Query("Vendor_Risk_Score", description=f"One of: {', '.join(sorted(ALLOWED_SORT_COLUMNS))}"),
    sort_desc: bool = True,
):
    if sort_by not in ALLOWED_SORT_COLUMNS:
        raise HTTPException(400, f"Unknown sort_by column: {sort_by!r}")

    df = data_store.vendors
    if search:
        df = df[df["Vendor_Name"].str.casefold().str.contains(search.casefold(), na=False)]
    if min_risk_score is not None:
        df = df[df["Vendor_Risk_Score"] >= min_risk_score]

    df = df.sort_values(sort_by, ascending=not sort_desc, na_position="last")
    total, page = paginate(df, skip, limit)
    return {"total": total, "skip": skip, "limit": limit, "items": to_records(page)}


@router.get("/{vendor_name}", summary="Vendor track record plus a sample of their recent works")
def get_vendor(vendor_name: str, works_limit: int = Query(20, ge=0, le=200)):
    df = data_store.vendors
    match = df[df["Vendor_Name"].str.casefold() == vendor_name.casefold()]
    if match.empty:
        raise HTTPException(404, f"Vendor {vendor_name!r} not found")
    detail = to_records(match)[0]

    works = data_store.works
    vendor_works = works[works["Vendor_Name"].str.casefold() == vendor_name.casefold()]
    sample = vendor_works[["Work_ID", "Work_Status", "State", "Constituency", "Sanction_Amount", "Cost_Overrun_Ratio"]]
    detail["works_count"] = int(len(vendor_works))
    detail["works_sample"] = to_records(sample.head(works_limit))
    return detail
