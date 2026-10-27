"""Differentiable candidate likelihood scoring for supervised training."""

from typing import Any

from .prompting import render_prompt
from .training import DecisionTrainingExample


class DifferentiableCandidateScorer:
    """Score numbered candidate continuations while preserving model gradients.

    The scorer uses the same prompt and candidate suffixes as inference scoring.
    It returns one summed token log-likelihood per candidate, in input order, and
    the index of the labeled candidate. The model and tokenizer are injected so
    callers control loading and tests can use small local doubles.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def score(self, example: DecisionTrainingExample) -> tuple[Any, int]:
        """Return differentiable candidate log scores and the target index."""
        torch = _torch()
        candidates = [(candidate.id, candidate.description) for candidate in example.candidates]
        prompt = render_prompt(example.context, candidates)
        device = self._model_device()

        scores = [
            self._suffix_log_likelihood(prompt, f"{index}]", device, torch)
            for index in range(1, len(example.candidates) + 1)
        ]

        try:
            target_index = next(
                index
                for index, candidate in enumerate(example.candidates)
                if candidate.id == example.target_id
            )
        except StopIteration as exc:
            raise ValueError("target_id must identify one of the candidates") from exc

        return torch.stack(scores), target_index

    def _suffix_log_likelihood(self, prompt: str, suffix: str, device: Any, torch: Any) -> Any:
        prompt_ids = self._encode(prompt)
        full_ids = self._encode(prompt + suffix)
        if not prompt_ids:
            raise ValueError("tokenizer produced no prompt tokens; cannot score candidate")
        if len(full_ids) <= len(prompt_ids):
            raise ValueError("tokenizer produced no suffix tokens; cannot score candidate")
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "tokenizer merged across the prompt/suffix boundary; cannot safely score candidate"
            )

        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        output = self.model(input_ids)
        logits = output.logits
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("causal model logits must have shape [1, sequence, vocabulary]")

        suffix_start = len(prompt_ids)
        target_ids = torch.tensor(full_ids[suffix_start:], dtype=torch.long, device=logits.device)
        next_token_logits = logits[0, suffix_start - 1 : len(full_ids) - 1, :]
        if next_token_logits.shape[0] != target_ids.shape[0]:
            raise ValueError("causal model logits do not cover all candidate suffix tokens")

        token_log_probs = torch.log_softmax(next_token_logits, dim=-1)
        return token_log_probs.gather(1, target_ids.unsqueeze(1)).sum()

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return [int(token_id) for token_id in input_ids]

    def _model_device(self) -> Any:
        """Use the model's parameter device when available; default doubles to CPU."""
        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            parameter = next(iter(parameters()), None)
            if parameter is not None:
                return parameter.device
        return getattr(self.model, "device", "cpu")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("differentiable scoring requires the 'training' extra") from exc
    return torch
