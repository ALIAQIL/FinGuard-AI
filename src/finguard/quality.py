"""Data-quality checks for the public ULB dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from finguard.config import ULB_COLUMNS, ULB_TARGET


@dataclass(frozen=True)
class QualityReport:
    rows: int
    columns: int
    missing_cells: int
    exact_duplicates: int
    invalid_target_rows: int
    negative_amount_rows: int
    schema_ok: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "missing_cells": self.missing_cells,
            "exact_duplicates": self.exact_duplicates,
            "invalid_target_rows": self.invalid_target_rows,
            "negative_amount_rows": self.negative_amount_rows,
            "schema_ok": self.schema_ok,
            "contract_valid_rows": self.rows - self.invalid_target_rows - self.negative_amount_rows,
        }


def validate_ulb_frame(frame: pd.DataFrame) -> QualityReport:
    """Return deterministic quality metrics without mutating the input."""
    missing = [column for column in ULB_COLUMNS if column not in frame.columns]
    invalid_target = 0
    negative_amount = 0
    if ULB_TARGET in frame:
        target = pd.to_numeric(frame[ULB_TARGET], errors="coerce")
        invalid_target = int((target.isna() | ~target.isin([0, 1])).sum())
    if "Amount" in frame:
        amount = pd.to_numeric(frame["Amount"], errors="coerce")
        negative_amount = int((amount < 0).sum())
    return QualityReport(
        rows=len(frame),
        columns=len(frame.columns),
        missing_cells=int(frame.isna().sum().sum()),
        exact_duplicates=int(frame.duplicated().sum()),
        invalid_target_rows=invalid_target,
        negative_amount_rows=negative_amount,
        schema_ok=not missing and set(frame.columns) == set(ULB_COLUMNS),
    )


def clean_ulb_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """Return accepted rows, quarantined rows, and the pre-clean report.

    Exact duplicates are removed from the accepted output, while invalid rows
    are retained in quarantine with a machine-readable reason.
    """
    report = validate_ulb_frame(frame)
    working = frame.copy()
    reasons = pd.Series("", index=working.index, dtype="string")
    target = pd.to_numeric(working[ULB_TARGET], errors="coerce")
    amount = pd.to_numeric(working["Amount"], errors="coerce")
    invalid = target.isna() | ~target.isin([0, 1]) | amount.isna() | (amount < 0)
    reasons.loc[target.isna() | ~target.isin([0, 1])] = "invalid_class"
    reasons.loc[amount.isna()] = "invalid_amount"
    reasons.loc[amount < 0] = "negative_amount"
    quarantine = working.loc[invalid].copy()
    if not quarantine.empty:
        quarantine["quarantine_reason"] = reasons.loc[quarantine.index].astype(str)
    accepted = working.loc[~invalid].drop_duplicates().copy()
    accepted[ULB_TARGET] = pd.to_numeric(accepted[ULB_TARGET]).astype("int8")
    accepted["Amount"] = pd.to_numeric(accepted["Amount"]).astype("float32")
    return accepted, quarantine, report
