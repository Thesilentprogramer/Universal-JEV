"""Reproducible configuration and manifests for local training experiments."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .prompting import TEMPLATE_VERSION


@dataclass(frozen=True)
class TrainingRunConfig:
    """Serializable settings required to reproduce a training run."""

    base_model: str
    dataset_path: str
    output_dir: str
    model_revision: str | None = None
    seed: int = 42
    learning_rate: float = 2e-4
    epochs: int = 1
    kl_weight: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    def __post_init__(self) -> None:
        if not self.base_model.strip():
            raise ValueError("base_model must not be empty")
        if self.seed < 0:
            raise ValueError("seed must not be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if self.epochs < 1:
            raise ValueError("epochs must be at least one")
        if self.kl_weight < 0:
            raise ValueError("kl_weight must not be negative")
        if self.lora_rank < 1 or self.lora_alpha < 1:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")


def build_run_manifest(config: TrainingRunConfig) -> dict[str, Any]:
    """Build a manifest with immutable inputs and experiment provenance."""
    dataset = Path(config.dataset_path)
    if not dataset.is_file():
        raise ValueError(f"dataset does not exist: {dataset}")
    return {
        "schema_version": 1,
        "template_version": TEMPLATE_VERSION,
        "dataset_sha256": sha256(dataset.read_bytes()).hexdigest(),
        "config": asdict(config),
    }


def write_run_manifest(config: TrainingRunConfig) -> Path:
    """Write a stable JSON manifest into the configured output directory."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run-manifest.json"
    path.write_text(
        json.dumps(build_run_manifest(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
