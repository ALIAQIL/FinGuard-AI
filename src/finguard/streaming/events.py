"""Validated event contract for replaying the anonymized ULB data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from finguard.config import ULB_FEATURE_COLUMNS


@dataclass(frozen=True)
class TransactionEvent:
    event_id: str
    event_timestamp: str
    features: dict[str, float]
    schema_version: str = "1"

    def validate(self) -> None:
        missing = set(ULB_FEATURE_COLUMNS) - set(self.features)
        if missing:
            raise ValueError(f"missing model fields: {sorted(missing)}")
        datetime.fromisoformat(self.event_timestamp.replace("Z", "+00:00"))
        if not self.event_id:
            raise ValueError("event_id is required")

    def to_json(self) -> str:
        self.validate()
        return json.dumps({
            "event_id": self.event_id,
            "event_timestamp": self.event_timestamp,
            "schema_version": self.schema_version,
            "features": self.features,
        }, sort_keys=True)


def event_from_row(row: dict[str, Any], source_row_number: int) -> TransactionEvent:
    """Build a stable event ID from the source row and its source position."""
    canonical = json.dumps({key: row.get(key) for key in ULB_FEATURE_COLUMNS}, sort_keys=True, default=str)
    event_id = hashlib.sha256(f"{source_row_number}:{canonical}".encode()).hexdigest()[:24]
    seconds = float(row["Time"])
    timestamp = datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
    event = TransactionEvent(
        event_id=event_id,
        event_timestamp=timestamp,
        features={key: float(row[key]) for key in ULB_FEATURE_COLUMNS},
    )
    event.validate()
    return event

