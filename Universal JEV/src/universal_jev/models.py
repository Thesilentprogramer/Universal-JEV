"""Public domain models for decision requests and results."""

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Candidate:
    id: str
    description: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("candidate id must not be empty")
        if not self.description.strip():
            raise ValueError("candidate description must not be empty")


@dataclass(frozen=True)
class DecisionRequest:
    context: str
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        if not self.context.strip():
            raise ValueError("decision context must not be empty")
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")


@dataclass(frozen=True)
class DecisionResult:
    selected_id: str | None
    abstained: bool
    log_scores: Mapping[str, float]
    normalized_scores: Mapping[str, float]
    reason: str | None = None
    metadata: Mapping[str, str] | None = None
