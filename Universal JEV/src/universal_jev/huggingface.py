"""Exact token-likelihood scoring for local causal Hugging Face models."""

from dataclasses import dataclass
from typing import Any

from .models import Candidate, DecisionRequest
from .prompting import TEMPLATE_VERSION, render_prompt


@dataclass(frozen=True)
class ScoringMetadata:
    method: str
    model: str
    tokenizer: str
    template_version: str = TEMPLATE_VERSION


class HuggingFaceExactScorer:
    """Score numbered candidate suffixes using a causal LM's next-token logits.

    The model and tokenizer are injected to keep loading policy outside this class and
    to make the scorer straightforward to test with lightweight doubles.
    """

    def __init__(self, model: Any, tokenizer: Any, *, model_name: str = "unknown") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name

    def score(self, request: DecisionRequest) -> tuple[dict[str, float], ScoringMetadata]:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError("exact Hugging Face scoring requires the 'local' extra") from exc

        candidates = [(candidate.id, candidate.description) for candidate in request.candidates]
        prompt = render_prompt(request.context, candidates)
        scores: dict[str, float] = {}
        for index, candidate in enumerate(request.candidates, 1):
            suffix = f"{index}]"
            scores[candidate.id] = self._suffix_log_likelihood(prompt, suffix, torch)
        metadata = ScoringMetadata(
            method="exact_token_log_likelihood",
            model=self.model_name,
            tokenizer=getattr(self.tokenizer, "name_or_path", "unknown"),
        )
        return scores, metadata

    def _suffix_log_likelihood(self, prompt: str, suffix: str, torch: Any) -> float:
        prompt_ids = self._encode(prompt)
        full_ids = self._encode(prompt + suffix)
        if len(full_ids) <= len(prompt_ids):
            raise ValueError("tokenizer produced no suffix tokens; cannot score candidate")
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError(
                "tokenizer merged across the prompt/suffix boundary; cannot safely score candidate"
            )
        suffix_start = len(prompt_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        with torch.no_grad():
            output = self.model(input_ids)
        logits = output.logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)
        total = 0.0
        for position in range(suffix_start, len(full_ids)):
            token_id = full_ids[position]
            total += float(log_probs[position - 1, token_id].item())
        return total

    def _encode(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        return list(input_ids)
