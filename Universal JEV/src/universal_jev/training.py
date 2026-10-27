"""Optional training primitives for paper-inspired decision fine-tuning.

This module intentionally provides composable, testable loss functions and dataset
contracts. Model loading, LoRA configuration, and experiment execution remain
explicit caller responsibilities.
"""

from dataclasses import dataclass
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .models import Candidate, DecisionRequest


@dataclass(frozen=True)
class DecisionTrainingExample:
    """A supervised decision with one correct member of the candidate set."""

    context: str
    candidates: tuple[Candidate, ...]
    target_id: str

    def __post_init__(self) -> None:
        request = DecisionRequest(self.context, self.candidates)
        if len(request.candidates) < 2:
            raise ValueError("training examples require at least two candidates")
        if self.target_id not in {candidate.id for candidate in request.candidates}:
            raise ValueError("target_id must identify one of the candidates")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DecisionTrainingExample":
        try:
            candidates = tuple(
                Candidate(id=candidate["id"], description=candidate["description"])
                for candidate in value["candidates"]
            )
            return cls(
                context=value["context"], candidates=candidates, target_id=value["target_id"]
            )
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid decision training example") from exc


def load_jsonl(path: str | Path) -> list[DecisionTrainingExample]:
    """Load labeled decisions from a UTF-8 JSON Lines file."""
    examples: list[DecisionTrainingExample] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            examples.append(DecisionTrainingExample.from_dict(json.loads(line)))
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"invalid training example on line {line_number}") from exc
    if not examples:
        raise ValueError("training data must contain at least one example")
    return examples


def listwise_loss(log_scores: Any, target_index: int) -> Any:
    """Negative log likelihood of the correct candidate under listwise scores."""
    torch = _torch()
    if log_scores.ndim != 1:
        raise ValueError("log_scores must be a one-dimensional tensor")
    if not 0 <= target_index < log_scores.shape[0]:
        raise ValueError("target_index must identify a candidate score")
    return torch.logsumexp(log_scores, dim=0) - log_scores[target_index]


def preservation_kl(current_log_probs: Any, reference_log_probs: Any) -> Any:
    """KL(reference || current) to retain reference-model behavior."""
    torch = _torch()
    if current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("current and reference log probabilities must share a shape")
    reference_probs = reference_log_probs.exp()
    return torch.sum(reference_probs * (reference_log_probs - current_log_probs))


def combined_loss(
    candidate_log_scores: Any,
    target_index: int,
    current_log_probs: Any | None = None,
    reference_log_probs: Any | None = None,
    *,
    kl_weight: float = 1.0,
) -> Any:
    """Combine decision supervision with optional reference-behavior preservation."""
    if kl_weight < 0:
        raise ValueError("kl_weight must not be negative")
    loss = listwise_loss(candidate_log_scores, target_index)
    if current_log_probs is None and reference_log_probs is None:
        return loss
    if current_log_probs is None or reference_log_probs is None:
        raise ValueError("both current and reference log probabilities are required for KL")
    return loss + kl_weight * preservation_kl(current_log_probs, reference_log_probs)


def prepare_lora_model(model: Any, *, target_modules: Iterable[str] = ("q_proj", "v_proj")) -> Any:
    """Wrap a causal model with a conservative LoRA configuration.

    PEFT remains optional so importing Universal JEV does not require training
    dependencies. The caller controls base-model loading and training arguments.
    """
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("LoRA preparation requires the 'training' extra") from exc
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=list(target_modules),
    )
    return get_peft_model(model, config)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("training losses require the 'local' extra") from exc
    return torch
