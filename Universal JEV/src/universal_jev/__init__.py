"""Universal JEV decision inference framework."""

__version__ = "0.1.0"

from .engine import DecisionEngine
from .huggingface import HuggingFaceExactScorer, ScoringMetadata
from .models import Candidate, DecisionRequest, DecisionResult
from .training import DecisionTrainingExample, combined_loss, listwise_loss, load_jsonl

__all__ = [
    "Candidate",
    "DecisionEngine",
    "DecisionRequest",
    "DecisionResult",
    "DecisionTrainingExample",
    "HuggingFaceExactScorer",
    "ScoringMetadata",
    "combined_loss",
    "listwise_loss",
    "load_jsonl",
]
