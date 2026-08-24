"""
Builds the SBERT semantic-duplicate-detection index used by the Hybrid Risk
Engine (Task 2 upgrade item: "duplicate detection").

Approach: embed every Work_Description with a sentence-transformer, then
within each (Vendor_Name, Constituency) group find each work's highest
cosine similarity to a *different* work by the same vendor in the same
place. High similarity there is the concrete fraud pattern this is meant to
catch: the same vendor billing near-identical work descriptions multiple
times in the same constituency (phantom/duplicate billing).

Comparisons are restricted to (Vendor_Name, Constituency) groups rather than
done globally for two reasons:
  1. Tractability: naive O(n^2) over 77k rows is ~5.9B pairs; within-group
     is a few hundred rows per group at most.
  2. Meaning: duplicate *billing* requires knowing who is billing twice.
     Vendor_Name == 'Unknown' is a placeholder for missing vendor identity,
     not a real shared identity, so those rows are excluded from grouping
     (duplicate_risk_score = 0, is_checked = False) rather than compared
     against each other.

Similarity -> risk scaling was calibrated empirically (see conversation /
README): within-group description similarity for legitimately different
works by the same vendor averages ~0.76 (std 0.10, 75th pct ~0.83) because
descriptions share templated phrasing. Only the extreme tail (>=0.85, close
to the observed max of ~0.96) is scored as risk, so ordinary templated
similarity doesn't get flagged - only near-identical descriptions do. This
signal is one of five blended inputs (20% weight), not a standalone verdict:
legitimately repeated micro-works (e.g. one solar light per household in a
village) can also produce high similarity, so a high score here should read
as "worth a human look," not "confirmed fraud."
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from feature_engineering import DATA_DIR

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
SIM_FLOOR = 0.85   # below this, duplicate_risk_score = 0
SIM_CEIL = 0.98    # at/above this, duplicate_risk_score = 100
MIN_GROUP_SIZE_FOR_BATCH_SIM = 2000  # guard against a single pathologically large group


def similarity_to_risk(sim: float) -> float:
    if sim <= SIM_FLOOR:
        return 0.0
    if sim >= SIM_CEIL:
        return 100.0
    return (sim - SIM_FLOOR) / (SIM_CEIL - SIM_FLOOR) * 100.0


def build_index():
    print("Loading works_master...")
    wm = pd.read_csv(os.path.join(DATA_DIR, "works_master.csv"),
                      usecols=["Work_ID", "Vendor_Name", "Constituency", "Work_Description"])
    wm["Work_Description"] = wm["Work_Description"].fillna("").astype(str)

    print(f"Loading SBERT model '{SBERT_MODEL_NAME}'...")
    model = SentenceTransformer(SBERT_MODEL_NAME)

    print(f"Encoding {len(wm)} work descriptions (this takes several minutes on CPU)...")
    t0 = time.time()
    embeddings = model.encode(wm["Work_Description"].tolist(), batch_size=128,
                               show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)
    print(f"Encoded in {time.time() - t0:.1f}s. Embedding shape: {embeddings.shape}")

    wm["duplicate_risk_score"] = 0.0
    wm["nearest_duplicate_work_id"] = None
    wm["nearest_similarity"] = 0.0
    wm["duplicate_checked"] = False

    real_vendor_mask = wm["Vendor_Name"].notna() & (wm["Vendor_Name"].str.strip().str.lower() != "unknown")
    groups = wm[real_vendor_mask].groupby(["Vendor_Name", "Constituency"]).indices

    print(f"Scoring {len(groups)} (vendor, constituency) groups for near-duplicate descriptions...")
    t0 = time.time()
    scored_groups = 0
    for (_vendor, _const), idx in groups.items():
        idx = np.asarray(idx)
        if len(idx) < 2:
            continue
        if len(idx) > MIN_GROUP_SIZE_FOR_BATCH_SIM:
            # extremely large group: sample to keep this tractable
            idx = np.random.RandomState(42).choice(idx, MIN_GROUP_SIZE_FOR_BATCH_SIM, replace=False)
        sub_emb = embeddings[idx]
        sim = cosine_similarity(sub_emb)
        np.fill_diagonal(sim, -1.0)  # exclude self
        best_j = sim.argmax(axis=1)
        best_sim = sim[np.arange(len(idx)), best_j]

        work_ids = wm["Work_ID"].values[idx]
        wm.loc[wm.index[idx], "duplicate_checked"] = True
        wm.loc[wm.index[idx], "nearest_similarity"] = best_sim
        wm.loc[wm.index[idx], "nearest_duplicate_work_id"] = work_ids[best_j]
        wm.loc[wm.index[idx], "duplicate_risk_score"] = [similarity_to_risk(s) for s in best_sim]
        scored_groups += 1

    print(f"Scored {scored_groups} groups in {time.time() - t0:.1f}s.")

    flagged = (wm["duplicate_risk_score"] > 0).sum()
    print(f"{flagged} works ({flagged / len(wm):.2%}) have a nonzero duplicate risk score "
          f"(similarity >= {SIM_FLOOR} to another work by the same vendor in the same constituency).")

    # Save the lookup table used at inference time for known Work_IDs
    out_path = os.path.join(MODEL_DIR, "duplicate_scores.csv")
    wm[["Work_ID", "Vendor_Name", "Constituency", "duplicate_risk_score",
        "nearest_duplicate_work_id", "nearest_similarity", "duplicate_checked"]].to_csv(out_path, index=False)
    print(f"Duplicate score lookup saved to {out_path}")

    # Save the raw embedding index so a brand-new (not-yet-in-dataset) project
    # description can be compared on the fly against its vendor+constituency
    # peers without re-encoding the whole dataset.
    index_path = os.path.join(MODEL_DIR, "doc_embeddings.pkl")
    with open(index_path, "wb") as f:
        pickle.dump({
            "model_name": SBERT_MODEL_NAME,
            "work_ids": wm["Work_ID"].values,
            "vendor_names": wm["Vendor_Name"].values,
            "constituencies": wm["Constituency"].values,
            "embeddings": embeddings,
        }, f)
    print(f"Embedding index saved to {index_path} ({embeddings.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    build_index()
