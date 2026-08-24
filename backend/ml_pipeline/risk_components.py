"""
Individual risk-signal scorers for the Hybrid Risk Engine. Every function
here takes a dataframe (batch, used at training/index-build time) or a
single-row dict (used at live-inference time in hybrid_risk_engine.py) and
returns a 0-100 score where HIGHER = RISKIER. Keeping these in one module
means the composite pseudo-label used to train the XGBoost model and the
live explanation shown by predict_project_risk() can never drift apart.
"""
import numpy as np
import pandas as pd

from feature_engineering import REQUIRED_DOCS, ISSUE_TOKENS, _split_tokens

# Weights used to blend the individual signals into one composite 0-100
# score. Must sum to 1.0. Tune here - both training and inference import
# these, so a reweight takes effect everywhere at once.
RISK_WEIGHTS = {
    "anomaly": 0.25,       # Isolation Forest (statistical outlier on cost/timeline/progress)
    "compliance": 0.20,    # rule-based: missing docs, NOC pending, low compliance score, issues
    "contractor": 0.20,    # vendor track record (risk score, monopoly, quality, thin history)
    "spatial": 0.15,       # geographic zone fraud risk
    "duplicate": 0.20,     # SBERT semantic duplicate-description detection
}
assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 1e-9


def _clip(x):
    return np.clip(x, 0, 100)


# ---------------------------------------------------------------------------
# Compliance rule score
# ---------------------------------------------------------------------------
def compliance_rule_score(row_or_df):
    """Deterministic, auditable rule score. Penalizes: pending NOC, missing
    required documents, logged issues, and a low Compliance_Score."""
    if isinstance(row_or_df, pd.DataFrame):
        df = row_or_df
        docs = df["Documents_Submitted"].apply(_split_tokens)
        issues = df["Issues_Encountered"].apply(_split_tokens)
        noc = df["NOC_Status"]
        compliance_score = df["Compliance_Score"]
    else:
        docs = pd.Series([_split_tokens(row_or_df.get("Documents_Submitted"))])
        issues = pd.Series([_split_tokens(row_or_df.get("Issues_Encountered"))])
        noc = pd.Series([row_or_df.get("NOC_Status")])
        compliance_score = pd.Series([row_or_df.get("Compliance_Score", 50.0)])

    missing_docs = docs.apply(lambda toks: sum(1 for d in REQUIRED_DOCS if d not in toks))
    doc_penalty = (missing_docs / len(REQUIRED_DOCS)) * 100

    noc_penalty = noc.map({"Pending": 100, "Not_Required": 0, "Received": 0}).fillna(40)

    issue_penalty = issues.apply(lambda toks: min(len(toks), 2) / 2 * 100)

    compliance_score = compliance_score.fillna(50.0)
    score_penalty = 100 - compliance_score.clip(0, 100)

    score = 0.35 * doc_penalty + 0.30 * noc_penalty + 0.15 * issue_penalty + 0.20 * score_penalty
    result = _clip(score)
    return result if isinstance(row_or_df, pd.DataFrame) else float(result.iloc[0])


# ---------------------------------------------------------------------------
# Contractor / vendor risk score
# ---------------------------------------------------------------------------
def contractor_risk_score(row_or_df):
    """Blends the vendor dimension's own risk score with monopoly
    concentration, (inverted) quality rating, and a thin-track-record
    penalty for vendors with few completed projects on file."""
    is_df = isinstance(row_or_df, pd.DataFrame)
    df = row_or_df if is_df else pd.DataFrame([row_or_df])

    vendor_risk = df.get("Vendor_Risk_Score", pd.Series(50.0, index=df.index)).fillna(50.0)
    monopoly = df.get("Vendor_Monopoly_Score", pd.Series(20.0, index=df.index)).fillna(20.0)
    monopoly_scaled = ((monopoly - 5) / (40 - 5) * 100).clip(0, 100)  # observed range ~5-40
    quality = df.get("Vendor_Quality_Rating", pd.Series(3.0, index=df.index)).fillna(3.0)
    quality_penalty = ((5 - quality) / 4 * 100).clip(0, 100)  # rating is 1(bad)-5(good)
    project_count = df.get("Vendor_Total_Projects_Count", pd.Series(5, index=df.index)).fillna(1)
    thin_history_penalty = np.where(project_count <= 1, 60, np.where(project_count <= 3, 25, 0))

    score = (0.45 * vendor_risk + 0.20 * monopoly_scaled + 0.20 * quality_penalty
             + 0.15 * thin_history_penalty)
    result = _clip(score)
    return result if is_df else float(result.iloc[0])


# ---------------------------------------------------------------------------
# Spatial risk score
# ---------------------------------------------------------------------------
def spatial_risk_score(row_or_df):
    """Uses the precomputed Zone_Fraud_Risk_Score from geography_dimension."""
    is_df = isinstance(row_or_df, pd.DataFrame)
    df = row_or_df if is_df else pd.DataFrame([row_or_df])
    zone = df.get("Zone_Fraud_Risk_Score", pd.Series(np.nan, index=df.index))
    zone = zone.fillna(zone.mean() if is_df and zone.notna().any() else 47.0)
    result = _clip(zone)
    return result if is_df else float(result.iloc[0])


# ---------------------------------------------------------------------------
# Anomaly score (Isolation Forest decision_function -> 0-100, higher=riskier)
# ---------------------------------------------------------------------------
def percentile_rank(value, sorted_reference: np.ndarray) -> float:
    """Where `value` falls in a sorted reference array, as 0..1."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or len(sorted_reference) == 0:
        return 0.5
    return float(np.searchsorted(sorted_reference, value) / len(sorted_reference))


def severity_from_score(score: float, critical_at: float = 70, warning_at: float = 40):
    """Maps a 0-100 sub-score to a (severity, emoji) pair for the WHY panel.
    Returns (None, None) if below the warning threshold - i.e. not flagged."""
    if score >= critical_at:
        return "critical", "\U0001F534"  # 🔴
    if score >= warning_at:
        return "warning", "\U0001F7E0"  # 🟠
    return None, None


def cost_anomaly_score(row: dict, per_feature_reference: dict) -> float:
    """0-100: how extreme this project's Cost_per_Unit / vendor cost-overrun
    history is vs the training distribution."""
    cost_pct = percentile_rank(row.get("Cost_per_Unit"), per_feature_reference["Cost_per_Unit"])
    overrun_pct = percentile_rank(row.get("Vendor_Avg_Cost_Overrun_Ratio"),
                                   per_feature_reference["Vendor_Avg_Cost_Overrun_Ratio"])
    return float(_clip(max(cost_pct, overrun_pct) * 100))


def timeline_anomaly_score(row: dict, per_feature_reference: dict) -> float:
    """0-100: how extreme this project's sanction speed / vendor timeline /
    disbursement-vs-progress gap is vs the training distribution."""
    sanction_pct = percentile_rank(row.get("Days_to_Sanction"), per_feature_reference["Days_to_Sanction"])
    vendor_timeline_pct = percentile_rank(row.get("Vendor_Avg_Timeline_Days"),
                                           per_feature_reference["Vendor_Avg_Timeline_Days"])
    progress_gap = row.get("Progress_Gap")
    # Progress_Gap reference is signed; extreme in either direction is anomalous,
    # so rank its absolute value against the absolute reference distribution.
    abs_ref = np.sort(np.abs(per_feature_reference["Progress_Gap"]))
    gap_pct = percentile_rank(abs(progress_gap) if progress_gap is not None and not pd.isna(progress_gap) else None,
                               abs_ref)
    return float(_clip(max(sanction_pct, vendor_timeline_pct, gap_pct) * 100))


def anomaly_scores_from_iforest(model, X_scaled):
    """Converts IsolationForest.score_samples (higher = more normal) into a
    0-100 risk scale (higher = more anomalous) via percentile rank, so the
    scale is stable and comparable to the other rule-based signals."""
    raw = model.score_samples(X_scaled)  # higher = more normal
    ranks = pd.Series(raw).rank(pct=True)  # 0..1, higher raw -> higher pct
    risk = (1 - ranks) * 100  # invert: most normal -> 0, most anomalous -> 100
    return risk.values
