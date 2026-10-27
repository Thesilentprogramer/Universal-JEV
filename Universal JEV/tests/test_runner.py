import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from universal_jev import Candidate
from universal_jev.experiment import TrainingRunConfig
from universal_jev.runner import run_training
from universal_jev.training import DecisionTrainingExample


class CharacterTokenizer:
    """Deterministic tokenizer where each character is a token."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(character) for character in text]}


def make_tiny_causal_model(torch, manifest_path=None):
    """Build a tiny causal model with trainable logits for the scorer."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 8)
            self.projection = torch.nn.Linear(8, 128)
            self.manifest_path = manifest_path

        def forward(self, input_ids):
            if self.manifest_path is not None:
                assert self.manifest_path.is_file()
            logits = self.projection(self.embedding(input_ids))
            return SimpleNamespace(logits=logits)

    return Model()


def _example() -> DecisionTrainingExample:
    return DecisionTrainingExample(
        context="Choose a tool for this task.",
        candidates=(
            Candidate("search", "Search the web"),
            Candidate("answer", "Answer directly"),
        ),
        target_id="search",
    )


def _config(tmp_path: Path, *, epochs=1) -> TrainingRunConfig:
    dataset_path = tmp_path / "training.jsonl"
    dataset_path.write_text('{"fixture": true}\n', encoding="utf-8")
    return TrainingRunConfig(
        base_model="injected-test-model",
        dataset_path=str(dataset_path),
        output_dir=str(tmp_path / "run"),
        epochs=epochs,
    )


def test_runner_performs_optimizer_steps_and_writes_manifest_before_forward(tmp_path):
    torch = pytest.importorskip("torch")
    config = _config(tmp_path, epochs=2)
    manifest_path = Path(config.output_dir) / "run-manifest.json"
    model = make_tiny_causal_model(torch, manifest_path)
    original_weight = model.projection.weight.detach().clone()

    summary = run_training(model, CharacterTokenizer(), [_example(), _example()], config)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert summary.step_count == 4
    assert summary.mean_loss > 0
    assert summary.manifest_path == manifest_path
    assert manifest["config"]["base_model"] == "injected-test-model"
    assert not torch.equal(original_weight, model.projection.weight.detach())
    assert summary.model_path is None
    assert not (Path(config.output_dir) / "model").exists()


def test_runner_saves_only_through_save_pretrained(tmp_path):
    torch = pytest.importorskip("torch")
    model = make_tiny_causal_model(torch)

    def save_pretrained(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "saved.txt").write_text("saved", encoding="utf-8")

    model.save_pretrained = save_pretrained
    config = _config(tmp_path)
    model_path = Path(config.output_dir) / "model"
    summary = run_training(model, CharacterTokenizer(), [_example()], config)

    assert summary.model_path == model_path
    assert (model_path / "saved.txt").read_text(encoding="utf-8") == "saved"


def test_runner_rejects_empty_examples(tmp_path):
    torch = pytest.importorskip("torch")
    config = _config(tmp_path)

    with pytest.raises(ValueError, match="must not be empty"):
        run_training(make_tiny_causal_model(torch), CharacterTokenizer(), [], config)


def test_runner_rejects_model_without_trainable_parameters(tmp_path):
    torch = pytest.importorskip("torch")

    class FrozenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)

    config = _config(tmp_path)
    with pytest.raises(ValueError, match="at least one trainable"):
        run_training(FrozenModel(), CharacterTokenizer(), [_example()], config)


def test_runner_rejects_kl_weight_until_preservation_data_is_supported(tmp_path):
    torch = pytest.importorskip("torch")
    base = _config(tmp_path)
    config = TrainingRunConfig(
        base_model=base.base_model,
        dataset_path=base.dataset_path,
        output_dir=base.output_dir,
        kl_weight=1.0,
    )

    with pytest.raises(ValueError, match="kl_weight must be zero"):
        run_training(make_tiny_causal_model(torch), CharacterTokenizer(), [_example()], config)


def test_runner_rejects_reusing_a_nonempty_output_directory(tmp_path):
    torch = pytest.importorskip("torch")
    config = _config(tmp_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True)
    old_model = output_dir / "model"
    old_model.mkdir()
    (old_model / "adapter.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ValueError, match="output_dir must be empty"):
        run_training(make_tiny_causal_model(torch), CharacterTokenizer(), [_example()], config)

    assert (old_model / "adapter.txt").read_text(encoding="utf-8") == "old"
    assert not (output_dir / "run-manifest.json").exists()


def test_runner_applies_config_seed_to_training_randomness(tmp_path):
    torch = pytest.importorskip("torch")

    class DropoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(128, 8)
            self.dropout = torch.nn.Dropout(0.5)
            self.projection = torch.nn.Linear(8, 128)

        def forward(self, input_ids):
            logits = self.projection(self.dropout(self.embedding(input_ids)))
            return SimpleNamespace(logits=logits)

    torch.manual_seed(7)
    first = DropoutModel()
    second = DropoutModel()
    second.load_state_dict(first.state_dict())
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    first_config = _config(tmp_path / "first")
    second_config = _config(tmp_path / "second")

    torch.manual_seed(111)
    first_summary = run_training(first, CharacterTokenizer(), [_example()], first_config)
    torch.manual_seed(999)
    second_summary = run_training(second, CharacterTokenizer(), [_example()], second_config)

    assert first_summary.mean_loss == pytest.approx(second_summary.mean_loss)
