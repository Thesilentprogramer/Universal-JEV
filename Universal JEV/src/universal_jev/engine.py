"""Decision engine orchestration for score-producing adapters."""

from collections.abc import Mapping

from .models import DecisionRequest, DecisionResult
from .policy import choose
from .scoring import softmax


class DecisionEngine:
    def decide(
        self,
        request: DecisionRequest,
        log_scores: Mapping[str, float],
        *,
        minimum_score: float = 0.0,
        minimum_margin: float = 0.0,
        metadata: Mapping[str, str] | None = None,
    ) -> DecisionResult:
        expected_ids = {candidate.id for candidate in request.candidates}
        if set(log_scores) != expected_ids:
            raise ValueError("scores must contain exactly one score for every candidate")
        normalized = softmax(log_scores)
        selected, reason = choose(
            normalized, minimum_score=minimum_score, minimum_margin=minimum_margin
        )
        return DecisionResult(
            selected_id=selected,
            abstained=selected is None,
            log_scores=dict(log_scores),
            normalized_scores=normalized,
            reason=reason,
            metadata=metadata,
        )
