from fastapi import APIRouter

from app.data_store import data_store

router = APIRouter()


@router.get("/overview", summary="Dashboard summary numbers across the whole dataset")
def overview():
    works = data_store.works
    mps = data_store.mps
    vendors = data_store.vendors
    geography = data_store.geography

    return {
        "total_works": int(len(works)),
        "total_mps": int(len(mps)),
        "total_vendors": int(len(vendors)),
        "total_constituencies": int(len(geography)),
        "total_recommended_amount": float(works["Recommended_Amount"].sum(skipna=True)),
        "total_sanctioned_amount": float(works["Sanction_Amount"].sum(skipna=True)),
        "total_disbursed_amount": float(works["Disbursed_Amount"].sum(skipna=True)),
        "avg_cost_overrun_ratio": round(float(works["Cost_Overrun_Ratio"].mean(skipna=True)), 3),
        "avg_days_to_sanction": round(float(works["Days_to_Sanction"].mean(skipna=True)), 1),
        "calamity_flagged_works": int(works["Calamity_Flag"].fillna(False).astype(bool).sum()),
        "works_by_status": works["Work_Status"].value_counts(dropna=True).to_dict(),
        "works_by_state": works["State"].value_counts(dropna=True).to_dict(),
        "works_by_house": works["House"].value_counts(dropna=True).to_dict(),
    }
