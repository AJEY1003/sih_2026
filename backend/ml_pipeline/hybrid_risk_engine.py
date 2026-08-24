"""
Hybrid Risk Engine - single entry point: predict_project_risk(project_data).

Ties together every signal built across Tasks 2 and 4:
  - Isolation Forest anomaly score      (train_task2_anomaly.py)
  - Compliance rule score               (risk_components.py)
  - Contractor / vendor risk score      (risk_components.py)
  - Spatial (zone) risk score           (risk_components.py)
  - SBERT semantic duplicate detection  (build_duplicate_index.py)
  - XGBoost cross-check score           (train_hybrid_xgboost.py)

The PRIMARY 0-100 final_risk_score is the weighted rule-based composite
(risk_components.RISK_WEIGHTS) - every point of it traces back to a named,
auditable signal, which is what "explainable" means here. The XGBoost score
is reported alongside as an independent, learned cross-check (and flagged
if it disagrees sharply with the rule composite), not blended into the
primary number - see train_hybrid_xgboost.py docstring for why.

Usage:
    from hybrid_risk_engine import predict_project_risk
    result = predict_project_risk({"Work_ID": "WS/MP620/2024-2025/133166"})
    # or, for a brand-new project not yet in the dataset, pass the raw fields
    # directly (Work_ID optional): Vendor_Name, Constituency, State,
    # Constituency_ID, Work_Description, Work_Status, Sanction_Amount,
    # Disbursed_Amount, Cost_per_Unit, Days_to_Sanction, Compliance_Score,
    # NOC_Status, Documents_Submitted, Issues_Encountered, ...
"""
import os
import pickle
import threading

import numpy as np
import pandas as pd

from feature_engineering import (
    DATA_DIR, build_master_dataset, add_progress_gap, add_document_flags,
    impute_vendor_features, build_xgb_feature_frame, align_xgb_features,
)
from risk_components import (
    RISK_WEIGHTS, compliance_rule_score, contractor_risk_score, spatial_risk_score,
    cost_anomaly_score, timeline_anomaly_score, severity_from_score,
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

RISK_LEVEL_BINS = [(75, "Critical"), (50, "High"), (25, "Medium"), (0, "Low")]
DUPLICATE_SIM_FLOOR = 0.85
DUPLICATE_SIM_CEIL = 0.98

_lock = threading.Lock()
_state = {}  # lazy-loaded singletons live here


def _risk_level(score: float) -> str:
    for threshold, label in RISK_LEVEL_BINS:
        if score >= threshold:
            return label
    return "Low"


def _load_state():
    """Loads every model artifact once, on first use."""
    with _lock:
        if _state.get("loaded"):
            return _state
        with open(os.path.join(MODEL_DIR, "anomaly_model.pkl"), "rb") as f:
            _state["if_model"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "anomaly_scaler.pkl"), "rb") as f:
            _state["if_scaler"] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "anomaly_meta.pkl"), "rb") as f:
            _state["if_meta"] = pickle.load(f)

        xgb_model_path = os.path.join(MODEL_DIR, "hybrid_xgboost_model.pkl")
        xgb_meta_path = os.path.join(MODEL_DIR, "hybrid_xgboost_meta.pkl")
        if os.path.exists(xgb_model_path):
            with open(xgb_model_path, "rb") as f:
                _state["xgb_model"] = pickle.load(f)
            with open(xgb_meta_path, "rb") as f:
                _state["xgb_meta"] = pickle.load(f)
        else:
            _state["xgb_model"] = None
            _state["xgb_meta"] = None

        dup_path = os.path.join(MODEL_DIR, "duplicate_scores.csv")
        _state["dup_lookup"] = pd.read_csv(dup_path).set_index("Work_ID") if os.path.exists(dup_path) else None

        dup_index_path = os.path.join(MODEL_DIR, "doc_embeddings.pkl")
        _state["dup_index"] = None  # loaded lazily below only if needed (large + needs torch)

        _state["master_df"] = None  # loaded lazily on first Work_ID lookup
        _state["loaded"] = True
        return _state


def _get_master_df():
    if _state.get("master_df") is None:
        _state["master_df"] = build_master_dataset()
    return _state["master_df"]


def _get_dup_index_and_model():
    if _state.get("dup_index") is None:
        dup_index_path = os.path.join(MODEL_DIR, "doc_embeddings.pkl")
        if not os.path.exists(dup_index_path):
            return None, None
        with open(dup_index_path, "rb") as f:
            _state["dup_index"] = pickle.load(f)
    if _state.get("sbert_model") is None:
        from sentence_transformers import SentenceTransformer
        _state["sbert_model"] = SentenceTransformer(_state["dup_index"]["model_name"])
    return _state["dup_index"], _state["sbert_model"]


def _resolve_row(project_data: dict) -> dict:
    """Starts from the full stored record for Work_ID (if found), then
    applies any fields explicitly passed in project_data on top - so
    callers can both look up a known work and run "what if I change X"
    scenarios."""
    row = {}
    work_id = project_data.get("Work_ID")
    if work_id:
        df = _get_master_df()
        match = df[df["Work_ID"] == work_id]
        if len(match):
            row = match.iloc[0].to_dict()
    row.update(project_data)
    return row


def _anomaly_score_for_row(row: dict, state: dict) -> float:
    meta = state["if_meta"]
    single_df = pd.DataFrame([row])
    single_df, _ = impute_vendor_features(single_df, meta["vendor_medians"])
    if "Progress_Gap" not in single_df.columns or pd.isna(single_df["Progress_Gap"].iloc[0]):
        if {"Disbursed_Amount", "Sanction_Amount", "Work_Status"}.issubset(single_df.columns):
            single_df = add_progress_gap(single_df)
        else:
            single_df["Progress_Gap"] = 0.0

    X = single_df.reindex(columns=meta["features"], fill_value=0).fillna(0)
    X_scaled = state["if_scaler"].transform(X)
    raw_score = state["if_model"].score_samples(X_scaled)[0]  # higher = more normal

    ref = meta["reference_scores"]
    percentile = np.searchsorted(ref, raw_score) / len(ref)
    return float(np.clip((1 - percentile) * 100, 0, 100))


def _duplicate_score_for_row(row: dict, state: dict) -> dict:
    work_id = row.get("Work_ID")
    if state["dup_lookup"] is not None and work_id in state["dup_lookup"].index:
        rec = state["dup_lookup"].loc[work_id]
        if isinstance(rec, pd.DataFrame):  # duplicate Work_ID edge case
            rec = rec.iloc[0]
        return {
            "score": float(rec["duplicate_risk_score"]),
            "checked": bool(rec["duplicate_checked"]),
            "nearest_duplicate_work_id": rec.get("nearest_duplicate_work_id"),
            "nearest_similarity": float(rec.get("nearest_similarity", 0.0)),
        }

    # Not a known Work_ID - compute on the fly if we have enough to compare against.
    vendor = row.get("Vendor_Name")
    constituency = row.get("Constituency")
    description = row.get("Work_Description", "")
    if not vendor or str(vendor).strip().lower() == "unknown" or not description:
        return {"score": 0.0, "checked": False, "nearest_duplicate_work_id": None, "nearest_similarity": 0.0}

    dup_index, sbert_model = _get_dup_index_and_model()
    if dup_index is None:
        return {"score": 0.0, "checked": False, "nearest_duplicate_work_id": None, "nearest_similarity": 0.0}

    mask = (dup_index["vendor_names"] == vendor) & (dup_index["constituencies"] == constituency)
    if not mask.any():
        return {"score": 0.0, "checked": False, "nearest_duplicate_work_id": None, "nearest_similarity": 0.0}

    from sklearn.metrics.pairwise import cosine_similarity
    query_emb = sbert_model.encode([description], convert_to_numpy=True)
    peer_emb = dup_index["embeddings"][mask]
    peer_ids = dup_index["work_ids"][mask]
    sims = cosine_similarity(query_emb, peer_emb)[0]
    best_j = int(np.argmax(sims))
    best_sim = float(sims[best_j])
    if best_sim <= DUPLICATE_SIM_FLOOR:
        score = 0.0
    elif best_sim >= DUPLICATE_SIM_CEIL:
        score = 100.0
    else:
        score = (best_sim - DUPLICATE_SIM_FLOOR) / (DUPLICATE_SIM_CEIL - DUPLICATE_SIM_FLOOR) * 100.0
    return {"score": score, "checked": True, "nearest_duplicate_work_id": peer_ids[best_j],
            "nearest_similarity": best_sim}


def _xgboost_score_for_row(row: dict, state: dict) -> float:
    if state["xgb_model"] is None:
        return None
    meta = state["xgb_meta"]
    single_df = pd.DataFrame([row])
    single_df = add_document_flags(single_df) if "Documents_Submitted" in single_df.columns else single_df
    X, _ = build_xgb_feature_frame(single_df, medians=meta["vendor_medians"])
    X = align_xgb_features(X, meta["columns"])
    pred = state["xgb_model"].predict(X)[0]
    return float(np.clip(pred, 0, 100))


def _explain(components: dict, dup_info: dict) -> list:
    reasons = []
    if components["compliance"] > 40:
        reasons.append(("compliance", components["compliance"],
                         "Compliance gaps: missing required documents, pending NOC, and/or a low compliance score."))
    if components["contractor"] > 40:
        reasons.append(("contractor", components["contractor"],
                         "Vendor track record risk: high vendor risk score, monopoly concentration, "
                         "low quality rating, or thin project history."))
    if components["anomaly"] > 60:
        reasons.append(("anomaly", components["anomaly"],
                         "Statistical outlier vs peers on cost-per-unit, timeline, or disbursement-vs-progress profile."))
    if components["spatial"] > 60:
        reasons.append(("spatial", components["spatial"],
                         "Located in a geographic zone with an elevated historical fraud-risk score."))
    if dup_info["checked"] and dup_info["score"] > 0:
        reasons.append(("duplicate", dup_info["score"],
                         f"Description is highly similar (cosine similarity {dup_info['nearest_similarity']:.2f}) "
                         f"to work {dup_info['nearest_duplicate_work_id']} by the same vendor in the same "
                         f"constituency - possible duplicate/phantom billing, but also consistent with "
                         f"legitimately repeated small-scale works; recommend manual review."))
    reasons.sort(key=lambda r: r[1], reverse=True)
    return [r[2] for r in reasons[:3]] if reasons else ["No individual signal exceeded its risk threshold."]


def predict_project_risk(project_data: dict) -> dict:
    """Scores one project 0-100 (higher = riskier) with a full component
    breakdown. `project_data` needs at least {"Work_ID": ...} for a known
    project, or a dict of raw fields for a new one (see module docstring)."""
    state = _load_state()
    row = _resolve_row(project_data)

    anomaly = _anomaly_score_for_row(row, state)
    compliance = compliance_rule_score(row)
    contractor = contractor_risk_score(row)
    spatial = spatial_risk_score(row)
    dup_info = _duplicate_score_for_row(row, state)
    duplicate = dup_info["score"]

    components = {"anomaly": anomaly, "compliance": compliance, "contractor": contractor,
                  "spatial": spatial, "duplicate": duplicate}

    final_score = sum(RISK_WEIGHTS[k] * v for k, v in components.items())
    final_score = float(np.clip(final_score, 0, 100))

    xgb_score = _xgboost_score_for_row(row, state)

    result = {
        "work_id": row.get("Work_ID"),
        "final_risk_score": round(final_score, 2),
        "risk_level": _risk_level(final_score),
        "components": {k: round(v, 2) for k, v in components.items()},
        "component_weights": RISK_WEIGHTS,
        "duplicate_detail": {
            "checked": dup_info["checked"],
            "nearest_duplicate_work_id": dup_info["nearest_duplicate_work_id"],
            "nearest_similarity": round(dup_info["nearest_similarity"], 3),
        },
        "xgboost_cross_check_score": round(xgb_score, 2) if xgb_score is not None else None,
        "top_risk_factors": _explain(components, dup_info),
    }

    if xgb_score is not None and abs(xgb_score - final_score) > 25:
        result["top_risk_factors"].append(
            f"Note: the XGBoost cross-check model ({xgb_score:.1f}) disagrees notably with the "
            f"rule-based composite ({final_score:.1f}) - this project's feature combination is "
            f"unusual relative to historical patterns and may warrant a closer look."
        )

    return result


# ---------------------------------------------------------------------------
# Investigation report - assembles the GIS "click a project -> why is it
# risky -> what else should I look at" flow into one payload.
# ---------------------------------------------------------------------------
RECOMMENDED_ACTIONS = {
    "Critical": "AUDIT / FIELD VERIFICATION",
    "High": "Priority review by compliance officer",
    "Medium": "Routine monitoring",
    "Low": "No action required",
}

NEARBY_WORKS_TOP_N = 5


def _work_id_to_emb_index(state: dict) -> dict:
    if state.get("dup_id_to_idx") is None:
        dup_index, _ = _get_dup_index_and_model()
        if dup_index is None:
            state["dup_id_to_idx"] = {}
        else:
            state["dup_id_to_idx"] = {wid: i for i, wid in enumerate(dup_index["work_ids"])}
    return state["dup_id_to_idx"]


def _nearby_similar_works(row: dict, state: dict, top_n: int = NEARBY_WORKS_TOP_N) -> list:
    """Other works in the same spatial cluster (falling back to the same
    constituency), ranked by SBERT description similarity - "what else near
    here looks like this project"."""
    dup_index, _ = _get_dup_index_and_model()
    if dup_index is None:
        return []
    id_to_idx = _work_id_to_emb_index(state)
    work_id = row.get("Work_ID")
    q_idx = id_to_idx.get(work_id)
    if q_idx is None:
        return []
    query_emb = dup_index["embeddings"][q_idx]

    df = _get_master_df()
    cluster_id = row.get("Spatial_Cluster_ID")
    if pd.notna(cluster_id):
        candidates = df[(df["Spatial_Cluster_ID"] == cluster_id) & (df["Work_ID"] != work_id)]
    else:
        candidates = df[(df["Constituency"] == row.get("Constituency")) & (df["Work_ID"] != work_id)]
    if candidates.empty:
        return []

    cand_ids = candidates["Work_ID"].values
    cand_idx = [id_to_idx[cid] for cid in cand_ids if cid in id_to_idx]
    if not cand_idx:
        return []
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity([query_emb], dup_index["embeddings"][cand_idx])[0]

    order = np.argsort(sims)[::-1][:top_n]
    results = []
    id_list = [cid for cid in cand_ids if cid in id_to_idx]
    lookup = candidates.set_index("Work_ID")
    for i in order:
        wid = id_list[i]
        rec = lookup.loc[wid]
        if isinstance(rec, pd.DataFrame):
            rec = rec.iloc[0]
        results.append({
            "work_id": wid,
            "vendor_name": rec.get("Vendor_Name"),
            "work_status": rec.get("Work_Status"),
            "cost_per_unit": None if pd.isna(rec.get("Cost_per_Unit")) else round(float(rec.get("Cost_per_Unit")), 2),
            "similarity": round(float(sims[i]), 3),
        })
    return results


def _contractor_history(row: dict, state: dict) -> dict:
    vendor = row.get("Vendor_Name")
    history = {
        "vendor_name": vendor,
        "total_projects": row.get("Vendor_Total_Projects_Count"),
        "avg_timeline_days": row.get("Vendor_Avg_Timeline_Days"),
        "avg_cost_overrun_ratio": row.get("Vendor_Avg_Cost_Overrun_Ratio"),
        "quality_rating": row.get("Vendor_Quality_Rating"),
        "monopoly_score": row.get("Vendor_Monopoly_Score"),
        "vendor_risk_score": row.get("Vendor_Risk_Score"),
        "other_flagged_duplicate_count": None,
    }
    if state["dup_lookup"] is not None and vendor and str(vendor).strip().lower() != "unknown":
        vendor_rows = state["dup_lookup"][state["dup_lookup"]["Vendor_Name"] == vendor]
        history["other_flagged_duplicate_count"] = int((vendor_rows["duplicate_risk_score"] > 0).sum())
    return {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in history.items()}


def _historical_benchmark(row: dict) -> dict:
    df = _get_master_df()
    unit = row.get("Work_Scope_Unit")
    cost = row.get("Cost_per_Unit")
    peers = df[df["Work_Scope_Unit"] == unit] if unit else df
    peer_median = peers["Cost_per_Unit"].median() if len(peers) else None
    ratio = (cost / peer_median) if (cost and peer_median) else None
    percentile = float((peers["Cost_per_Unit"] < cost).mean()) if (cost is not None and len(peers)) else None
    return {
        "work_scope_unit": unit,
        "peer_group_size": int(len(peers)),
        "peer_median_cost_per_unit": None if peer_median is None or pd.isna(peer_median) else round(float(peer_median), 2),
        "this_cost_per_unit": None if cost is None or pd.isna(cost) else round(float(cost), 2),
        "cost_vs_peer_median_ratio": None if ratio is None or pd.isna(ratio) else round(float(ratio), 2),
        "cost_percentile_vs_peers": None if percentile is None else round(percentile, 3),
    }


def generate_investigation_report(work_id: str) -> dict:
    """The backend payload for the GIS 'select project -> why is it risky ->
    what else should I check' investigation flow:

        Select project -> risk score & level -> WHY (color-coded factors)
        -> nearby similar works -> contractor history -> historical
        benchmark -> recommended action.

    Every field here is either a direct model/rule output or a simple,
    auditable aggregate over the underlying tables - nothing here is a new
    black box.
    """
    state = _load_state()
    row = _resolve_row({"Work_ID": work_id})
    if not row:
        raise ValueError(f"Unknown Work_ID: {work_id!r}")

    risk = predict_project_risk({"Work_ID": work_id})
    per_feature_ref = state["if_meta"]["per_feature_reference"]

    cost_score = cost_anomaly_score(row, per_feature_ref)
    timeline_score = timeline_anomaly_score(row, per_feature_ref)
    duplicate_score = risk["components"]["duplicate"]
    compliance_score = risk["components"]["compliance"]

    why_candidates = [
        ("Duplicate similarity", duplicate_score,
         f"Similarity {risk['duplicate_detail']['nearest_similarity']:.2f} to "
         f"{risk['duplicate_detail']['nearest_duplicate_work_id']}" if risk["duplicate_detail"]["checked"] else None),
        ("Cost anomaly", cost_score, "Cost-per-unit and/or vendor cost-overrun history is a statistical outlier."),
        ("Timeline anomaly", timeline_score, "Sanction speed, vendor timeline, or disbursement-vs-progress gap is a statistical outlier."),
        ("Compliance gaps", compliance_score, "Missing documents, pending NOC, logged issues, and/or a low compliance score."),
    ]
    why = []
    for label, score, detail in why_candidates:
        severity, emoji = severity_from_score(score)
        if severity is None:
            continue
        why.append({"label": label, "severity": severity, "icon": emoji, "score": round(score, 2), "detail": detail})
    why.sort(key=lambda w: w["score"], reverse=True)

    return {
        "work_id": work_id,
        "risk_score": risk["final_risk_score"],
        "risk_level": risk["risk_level"],
        "why": why,
        "nearby_similar_works": _nearby_similar_works(row, state),
        "contractor_history": _contractor_history(row, state),
        "historical_benchmark": _historical_benchmark(row),
        "recommended_action": RECOMMENDED_ACTIONS[risk["risk_level"]],
        "xgboost_cross_check_score": risk["xgboost_cross_check_score"],
    }


if __name__ == "__main__":
    df = build_master_dataset()
    sample_id = df["Work_ID"].iloc[0]
    print(f"Scoring sample project: {sample_id}")
    import json
    print(json.dumps(predict_project_risk({"Work_ID": sample_id}), indent=2, default=str))
