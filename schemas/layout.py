from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultsLayout:
    """On-disk layout for one prompt version's eval run. No database: just folders of JSON."""

    root: Path  # e.g. results/
    version: str  # e.g. v1

    @property
    def base(self) -> Path:
        return self.root / self.version

    @property
    def transcripts(self) -> Path:  # <call_id>.json  (Transcript)
        return self.base / "transcripts"

    @property
    def postcall(self) -> Path:  # <call_id>.json  (PostCallRecord)
        return self.base / "postcall"

    @property
    def scores(self) -> Path:  # <call_id>.json  (CallScore)
        return self.base / "scores"

    @property
    def summary(self) -> Path:  # Summary
        return self.base / "summary.json"

    @property
    def sim_state(self) -> Path:  # simulator resume state (owned by sim/)
        return self.base / "sim_state.json"

    def comparison(self, other: str) -> Path:  # Comparison
        return self.root / f"compare_{self.version}_{other}.json"
