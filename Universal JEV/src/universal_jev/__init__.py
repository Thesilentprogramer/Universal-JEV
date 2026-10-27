"""Universal JEV decision inference framework."""

__version__ = "0.1.0"

from .engine import DecisionEngine
from .experiment import TrainingRunConfig
from .huggingface import HuggingFaceExactScorer, ScoringMetadata
from .models import Candidate, DecisionRequest, DecisionResult
from .runner import TrainingSummary, run_training
from .training import DecisionTrainingExample, combined_loss, listwise_loss, load_jsonl
from .training_scorer import DifferentiableCandidateScorer

__all__ = [
    "Candidate",
    "DecisionEngine",
    "DecisionRequest",
    "DecisionResult",
    "DecisionTrainingExample",
    "DifferentiableCandidateScorer",
    "HuggingFaceExactScorer",
    "ScoringMetadata",
    "TrainingRunConfig",
    "TrainingSummary",
    "combined_loss",
    "listwise_loss",
    "load_jsonl",
    "run_training",
]
