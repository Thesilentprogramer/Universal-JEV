"""Universal JEV decision inference framework."""

__version__ = "0.1.0"

from .engine import DecisionEngine
from .models import Candidate, DecisionRequest, DecisionResult

__all__ = ["Candidate", "DecisionEngine", "DecisionRequest", "DecisionResult"]
