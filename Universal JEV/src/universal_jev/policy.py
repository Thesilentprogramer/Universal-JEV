"""Decision policies applied after model scoring."""

from collections.abc import Mapping


def choose(
    normalized_scores: Mapping[str, float],
    *,
    minimum_score: float = 0.0,
    minimum_margin: float = 0.0,
) -> tuple[str | None, str | None]:
    if not normalized_scores:
        raise ValueError("at least one normalized score is required")
    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be between zero and one")
    if minimum_margin < 0:
        raise ValueError("minimum_margin must not be negative")
    ranked = sorted(normalized_scores.items(), key=lambda item: (-item[1], item[0]))
    winner, winner_score = ranked[0]
    runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
    if winner_score < minimum_score:
        return None, "top_score_below_threshold"
    if winner_score - runner_score < minimum_margin:
        return None, "top_score_margin_below_threshold"
    return winner, None
