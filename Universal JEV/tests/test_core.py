import pytest

from universal_jev import Candidate, DecisionEngine, DecisionRequest
from universal_jev.prompting import CONTINUATION_PREFIX, render_prompt
from universal_jev.scoring import softmax


def request() -> DecisionRequest:
    return DecisionRequest(
        context="The user asks for current information.",
        candidates=(
            Candidate("search_web", "Search the web"),
            Candidate("answer_directly", "Answer directly"),
        ),
    )


def test_request_rejects_duplicate_ids():
    with pytest.raises(ValueError, match="unique"):
        DecisionRequest("context", (Candidate("same", "one"), Candidate("same", "two")))


def test_prompt_uses_stable_numbered_template():
    rendered = render_prompt("context", [("search_web", "Search the web")])
    assert "[1] Search the web" in rendered
    assert rendered.endswith(CONTINUATION_PREFIX)


def test_softmax_is_stable_and_normalized():
    scores = softmax({"a": -1000.0, "b": -999.0})
    assert sum(scores.values()) == pytest.approx(1.0)
    assert scores["b"] > scores["a"]


def test_engine_selects_best_candidate():
    result = DecisionEngine().decide(request(), {"search_web": 0.0, "answer_directly": -2.0})
    assert result.selected_id == "search_web"
    assert not result.abstained


def test_engine_abstains_when_margin_is_too_small():
    result = DecisionEngine().decide(
        request(), {"search_web": 0.0, "answer_directly": -0.01}, minimum_margin=0.5
    )
    assert result.abstained
    assert result.reason == "top_score_margin_below_threshold"
