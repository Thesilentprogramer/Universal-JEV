"""Offline-first sequential training runner for labeled decisions."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .experiment import TrainingRunConfig, write_run_manifest
from .training import DecisionTrainingExample, listwise_loss
from .training_scorer import DifferentiableCandidateScorer


@dataclass(frozen=True)
class TrainingSummary:
    """Basic results and artifact locations for a completed training run."""

    step_count: int
    mean_loss: float
    manifest_path: Path
    model_path: Path | None = None


def run_training(
    model: Any,
    tokenizer: Any,
    examples: Iterable[DecisionTrainingExample],
    config: TrainingRunConfig,
) -> TrainingSummary:
    """Train an injected model with one listwise example per optimizer step.

    The caller owns model and tokenizer construction. Examples are processed in
    input order, once per epoch. This runner applies listwise decision loss only;
    it does not load a model, download files, or compute a preservation objective.
    """
    torch = _torch()
    training_examples = tuple(examples)
    if not training_examples:
        raise ValueError("training examples must not be empty")
    if config.kl_weight != 0:
        raise ValueError(
            "this runner does not yet accept preservation data; kl_weight must be zero"
        )
    if any(not isinstance(example, DecisionTrainingExample) for example in training_examples):
        raise TypeError("examples must contain DecisionTrainingExample values")

    parameters_method = getattr(model, "parameters", None)
    if not callable(parameters_method):
        raise ValueError("model must expose trainable parameters")
    trainable_parameters = [
        parameter for parameter in parameters_method() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("model must have at least one trainable parameter")

    output_dir = Path(config.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_dir must be empty for a new training run")

    # This controls stochastic operations during optimization. Callers remain
    # responsible for constructing the injected model from reproducible weights.
    torch.manual_seed(config.seed)
    manifest_path = write_run_manifest(config)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.learning_rate)
    scorer = DifferentiableCandidateScorer(model, tokenizer)
    train_method = getattr(model, "train", None)
    if callable(train_method):
        train_method()

    losses: list[float] = []
    for _epoch in range(config.epochs):
        # Deliberately process one example at a time; no batching or padding is used.
        for example in training_examples:
            optimizer.zero_grad(set_to_none=True)
            candidate_scores, target_index = scorer.score(example)
            loss = listwise_loss(candidate_scores, target_index)
            losses.append(float(loss.detach().cpu().item()))
            loss.backward()
            optimizer.step()

    model_path: Path | None = None
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        model_path = Path(config.output_dir) / "model"
        save_pretrained(str(model_path))

    return TrainingSummary(
        step_count=len(losses),
        mean_loss=sum(losses) / len(losses),
        manifest_path=manifest_path,
        model_path=model_path,
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("training requires the 'training' extra") from exc
    return torch
