"""Pydantic request/response models for the Risk Engine routes.

Listing routes (works/mps/vendors/geography) return plain dicts built
straight from pandas rows - those tables are wide and mostly pass-through,
so a hand-typed schema per table would just duplicate the CSV headers with
no real validation value. The risk-scoring routes get real models because
that's the one place callers send a hand-built payload in.
"""
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ProjectRiskRequest(BaseModel):
    """Fields accepted by POST /api/risk/predict.

    Pass just `Work_ID` to score a known project as recorded in the
    dataset, or omit it and supply raw fields to score a brand-new project.
    Any fields given alongside `Work_ID` override the stored record for
    "what if I change X" scenarios. Extra fields beyond the ones listed
    here are passed straight through to the engine.
    """

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {"Work_ID": "WS/MP620/2024-2025/133166"},
                {
                    "Vendor_Name": "ABC Constructions",
                    "Constituency": "Example Constituency",
                    "State": "Example State",
                    "Constituency_ID": "MP620",
                    "Work_Description": "Construction of community hall",
                    "Work_Status": "Vendor Identification",
                    "Sanction_Amount": 500000,
                    "Disbursed_Amount": 480000,
                    "Cost_per_Unit": 999999.0,
                    "Days_to_Sanction": 2,
                    "Compliance_Score": 20.0,
                    "NOC_Status": "Pending",
                    "Documents_Submitted": "",
                    "Issues_Encountered": "Vendor issues",
                },
            ]
        },
    )

    Work_ID: Optional[str] = Field(None, description="Known Work_ID to look up and/or override.")
    Vendor_Name: Optional[str] = None
    Constituency: Optional[str] = None
    State: Optional[str] = None
    Constituency_ID: Optional[str] = None
    Work_Description: Optional[str] = None
    Work_Status: Optional[str] = None
    Sanction_Amount: Optional[float] = None
    Disbursed_Amount: Optional[float] = None
    Cost_per_Unit: Optional[float] = None
    Days_to_Sanction: Optional[float] = None
    Compliance_Score: Optional[float] = None
    NOC_Status: Optional[str] = None
    Documents_Submitted: Optional[str] = None
    Issues_Encountered: Optional[str] = None


class Page(BaseModel):
    """Generic pagination envelope used by every list endpoint."""

    total: int
    skip: int
    limit: int
    items: list[Any]
