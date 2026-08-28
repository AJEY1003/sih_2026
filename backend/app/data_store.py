"""
In-memory data access layer over the 5-table star schema produced by
`prepare_dataset.py` (data/output_schema/*.csv). CSVs are loaded once at
app startup and kept in memory - the largest table (works_master.csv) is
~77k rows, small enough to hold and filter/sort with pandas per-request.

The fully merged, feature-engineered dataframe (`.master`) reuses
`feature_engineering.build_master_dataset()` from the ML pipeline so a
single Work_ID lookup here is guaranteed to match what the Hybrid Risk
Engine sees - it's built lazily on first use since it's more expensive.
"""
import os
import threading

import pandas as pd

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(_BACKEND_DIR), "data", "output_schema")


def to_records(df: pd.DataFrame) -> list:
    """Row-wise dict conversion with NaN/NaT replaced by None, so the JSON
    response is clean (bare NaN is not valid JSON)."""
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


def paginate(df: pd.DataFrame, skip: int, limit: int):
    """Returns (total_row_count, page_slice)."""
    total = len(df)
    return total, df.iloc[skip: skip + limit]


class DataStore:
    """Lazily-populated singleton holding every dimension/fact table."""

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self.works: pd.DataFrame | None = None
        self.mps: pd.DataFrame | None = None
        self.vendors: pd.DataFrame | None = None
        self.geography: pd.DataFrame | None = None
        self.compliance: pd.DataFrame | None = None
        self._master: pd.DataFrame | None = None

    def load(self):
        """Eager-loads the raw dimension/fact CSVs. Call once at app startup."""
        with self._lock:
            if self._loaded:
                return
            self.works = pd.read_csv(os.path.join(DATA_DIR, "works_master.csv"))
            self.mps = pd.read_csv(os.path.join(DATA_DIR, "mp_dimension.csv"))
            self.vendors = pd.read_csv(os.path.join(DATA_DIR, "vendor_dimension.csv"))
            self.geography = pd.read_csv(os.path.join(DATA_DIR, "geography_dimension.csv"))
            self.compliance = pd.read_csv(os.path.join(DATA_DIR, "compliance_and_ml.csv"))
            self._loaded = True

    @property
    def master(self) -> pd.DataFrame:
        """Full merged dataframe (works + vendor + compliance + geography,
        with engineered columns) - built on first access, then cached."""
        if self._master is None:
            with self._lock:
                if self._master is None:
                    from feature_engineering import build_master_dataset
                    self._master = build_master_dataset()
        return self._master


data_store = DataStore()
