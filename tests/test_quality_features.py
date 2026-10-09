import pandas as pd

from finguard.config import ULB_COLUMNS
from finguard.features import add_ulb_features
from finguard.quality import clean_ulb_frame, validate_ulb_frame


def _frame():
    rows = []
    for i in range(31):
        rows.append({**{column: 0.0 for column in ULB_COLUMNS}, "Time": float(i * 3600), "Amount": float(i), "Class": i % 2})
    return pd.DataFrame(rows, columns=ULB_COLUMNS)


def test_quality_detects_duplicates_and_bad_amount():
    frame = _frame()
    frame.loc[0, "Amount"] = -1
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)
    report = validate_ulb_frame(frame)
    assert report.exact_duplicates == 1
    assert report.negative_amount_rows == 1
    accepted, quarantine, _ = clean_ulb_frame(frame)
    assert len(accepted) == 30
    assert len(quarantine) == 1
    assert quarantine.iloc[0]["quarantine_reason"] == "negative_amount"


def test_features_are_row_local_and_do_not_create_entity_history():
    result = add_ulb_features(_frame())
    assert {"event_hour", "event_day", "amount_log1p"}.issubset(result.columns)
    assert "customer_id" not in result.columns
    assert result.loc[1, "event_hour"] == 1

