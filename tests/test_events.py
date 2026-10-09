import json

from finguard.config import ULB_FEATURE_COLUMNS
from finguard.streaming.events import event_from_row


def test_event_serialization_is_valid_and_deterministic():
    row = {column: 1 for column in ULB_FEATURE_COLUMNS}
    first = event_from_row(row, 4)
    second = event_from_row(row, 4)
    assert first.event_id == second.event_id
    payload = json.loads(first.to_json())
    assert payload["event_id"] == first.event_id
    assert payload["schema_version"] == "1"

