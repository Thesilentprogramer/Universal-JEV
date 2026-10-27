import pytest

from universal_jev import Candidate
from universal_jev.training import DecisionTrainingExample
from universal_jev.training_scorer import DifferentiableCandidateScorer


class CharacterTokenizer:
    """A deterministic tokenizer where every character is one token."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def _training_example(candidates, target_id):
    return DecisionTrainingExample(
        context="choose a tool",
        candidates=tuple(candidates),
        target_id=target_id,
    )


def test_scores_sum_complete_suffix_likelihoods_and_keep_gradients():
    torch = pytest.importorskip("torch")

    class ConstantLogitModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = torch.nn.Parameter(torch.linspace(-0.5, 0.5, 128))

        def forward(self, input_ids):
            logits = self.bias.expand(input_ids.shape[0], input_ids.shape[1], -1)
            return type("Output", (), {"logits": logits})()

    candidates = tuple(Candidate(f"tool-{index}", f"Tool {index}") for index in range(1, 11))
    example = _training_example(candidates, target_id="tool-10")
    model = ConstantLogitModel()

    scores, target_index = DifferentiableCandidateScorer(model, CharacterTokenizer()).score(example)

    expected_token_log_probs = torch.log_softmax(model.bias, dim=-1)
    expected_ten_score = (
        expected_token_log_probs[ord("1")]
        + expected_token_log_probs[ord("0")]
        + expected_token_log_probs[ord("]")]
    )
    assert scores.shape == (10,)
    assert target_index == 9
    assert scores[9].item() == pytest.approx(expected_ten_score.item())
    assert scores.requires_grad

    scores.sum().backward()
    assert model.bias.grad is not None
    assert torch.count_nonzero(model.bias.grad).item() > 0


def test_target_index_tracks_candidate_order():
    pytest.importorskip("torch")

    class UniformModel:
        def __call__(self, input_ids):
            import torch

            logits = torch.zeros((1, input_ids.shape[1], 128))
            return type("Output", (), {"logits": logits})()

    candidates = (Candidate("first", "First"), Candidate("target", "Target"))
    example = _training_example(candidates, target_id="target")

    scores, target_index = DifferentiableCandidateScorer(
        UniformModel(), CharacterTokenizer()
    ).score(example)

    assert scores.shape == (2,)
    assert target_index == 1


def test_rejects_tokenizer_boundary_merge_before_model_call():
    class BoundaryMergingTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [9] if text.endswith("[") else [8, 1]}

    class UnusedModel:
        def __call__(self, input_ids):
            raise AssertionError("model must not run with unsafe token boundaries")

    example = _training_example(
        (Candidate("tool", "Tool"), Candidate("other", "Other")), target_id="tool"
    )
    scorer = DifferentiableCandidateScorer(UnusedModel(), BoundaryMergingTokenizer())

    with pytest.raises(ValueError, match="merged across"):
        scorer.score(example)
