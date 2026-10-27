import pytest

from universal_jev import Candidate, DecisionRequest, HuggingFaceExactScorer


class FakeTokenizer:
    name_or_path = "fake-tokenizer"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) % 23 for character in text]}


def test_scorer_assigns_numbered_suffixes(monkeypatch):
    pytest.importorskip("torch")

    class FakeOutput:
        logits = None

    class FakeModel:
        def __call__(self, input_ids):
            import torch

            # Uniform logits make the result deterministic; the test checks the contract.
            FakeOutput.logits = torch.zeros((1, input_ids.shape[1], 23))
            return FakeOutput()

    request = DecisionRequest(
        "choose a tool",
        (Candidate("first", "First tool"), Candidate("second", "Second tool")),
    )
    scores, metadata = HuggingFaceExactScorer(
        FakeModel(), FakeTokenizer(), model_name="fake-model"
    ).score(request)

    assert list(scores) == ["first", "second"]
    assert scores["first"] == pytest.approx(scores["second"])
    assert metadata.method == "exact_token_log_likelihood"
    assert metadata.model == "fake-model"


def test_scorer_rejects_tokenizer_boundary_merges():
    pytest.importorskip("torch")

    class BoundaryMergingTokenizer:
        def __call__(self, text, add_special_tokens=False):
            # A synthetic tokenizer where adding the suffix changes the final prompt token.
            return {"input_ids": [9] if text.endswith("[") else [8, 1]}

    class UnusedModel:
        def __call__(self, input_ids):
            raise AssertionError("model must not run with unsafe token boundaries")

    request = DecisionRequest("context", (Candidate("one", "One"),))
    with pytest.raises(ValueError, match="merged across"):
        HuggingFaceExactScorer(UnusedModel(), BoundaryMergingTokenizer()).score(request)
