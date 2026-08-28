"""
MPLADS Analytics & Monitoring API.

`backend/ml_pipeline` is a plain script directory (not an installable
package) whose modules import each other with bare top-level imports, e.g.
`from feature_engineering import ...` inside hybrid_risk_engine.py. To reuse
those modules from the FastAPI app without duplicating or rewriting them, we
add that directory to sys.path here - this runs exactly once, before any
submodule of `app` (routers, data_store) is imported, since Python always
executes a package's __init__.py before its submodules.
"""
import os
import sys

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML_PIPELINE_DIR = os.path.join(_BACKEND_DIR, "ml_pipeline")

if _ML_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _ML_PIPELINE_DIR)
