import json

import pytest

from universal_jev import Candidate, DecisionTrainingExample, combined_loss, listwise_loss, load_jsonl
from universal_jev.training import preservation_kl


def example() -> DecisionTrainingExample:
    return DecisionTrainingExample(
        context="Find current facts.",
        candidates=(Candidate("search", "Search the web"), Candidate("answer", "Answer directly")),
        target_id="search",
    )


def test_training_example_rejects_unknown_target():
    with pytest.raises(ValueError, match="target_id"):
        DecisionTrainingExample(
            "context", (Candidate("one", "One"), Candidate("two", "Two")), "missing"
        )


def test_training_example_requires_a_real_choice():
    with pytest.raises(ValueError, match="at least two"):
        DecisionTrainingExample("context", (Candidate("one", "One"),), "one")


def test_load_jsonl_reads_examples(tmp_path):
    path = tmp_path / "examples.jsonl"
    path.write_text(
        json.dumps(
            {
                "context": example().context,
                "candidates": [
                    {"id": "search", "description": "Search the web"},
                    {"id": "answer", "description": "Answer directly"},
                ],
                "target_id": "search",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_jsonl(path)[0].target_id == "search"


def test_listwise_loss_prefers_the_labeled_candidate():
    torch = pytest.importorskip("torch")
    good = listwise_loss(torch.tensor([3.0, 0.0]), 0)
    bad = listwise_loss(torch.tensor([0.0, 3.0]), 0)
    assert good < bad


def test_kl_is_zero_for_matching_distributions():
    torch = pytest.importorskip("torch")
    scores = torch.log_softmax(torch.tensor([1.0, 2.0]), dim=0)
    assert preservation_kl(scores, scores).item() == pytest.approx(0.0)


def test_combined_loss_requires_both_kl_inputs():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="both current"):
        combined_loss(torch.tensor([1.0, 0.0]), 0, current_log_probs=torch.tensor([0.0]))
