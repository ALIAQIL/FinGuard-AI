"""Leakage-safe features available in the ULB dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_ulb_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic temporal and amount features.

    The ULB data has no customer/card identifier, so no entity-history feature
    is fabricated. These transformations are row-local and can be reproduced
    during replay. Fitted transforms such as scalers are applied later on the
    training split only.
    """
    result = frame.copy()
    seconds = pd.to_numeric(result["Time"], errors="coerce")
    amount = pd.to_numeric(result["Amount"], errors="coerce")
    result["event_hour"] = ((seconds // 3600) % 24).astype("int8")
    result["event_day"] = (seconds // 86400).astype("int8")
    result["amount_log1p"] = np.log1p(amount.clip(lower=0)).astype("float32")
    return result
