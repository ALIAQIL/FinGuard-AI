"""Shared configuration and schema contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ULB_FEATURE_COLUMNS = ["Time", *(f"V{i}" for i in range(1, 29)), "Amount"]
ULB_TARGET = "Class"
ULB_COLUMNS = [*ULB_FEATURE_COLUMNS, ULB_TARGET]


@dataclass(frozen=True)
class Paths:
    """Filesystem locations used by local runs."""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

