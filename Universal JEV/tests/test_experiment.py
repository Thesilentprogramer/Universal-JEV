import json

import pytest

from universal_jev.experiment import TrainingRunConfig, build_run_manifest, write_run_manifest
from universal_jev.prompting import TEMPLATE_VERSION


def config(tmp_path) -> TrainingRunConfig:
    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"example": 1}\n', encoding="utf-8")
    return TrainingRunConfig(
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        model_revision="fixed-revision",
        dataset_path=str(dataset),
        output_dir=str(tmp_path / "output"),
    )


def test_manifest_records_dataset_hash_and_template(tmp_path):
    manifest = build_run_manifest(config(tmp_path))
    assert len(manifest["dataset_sha256"]) == 64
    assert manifest["template_version"] == TEMPLATE_VERSION
    assert manifest["config"]["model_revision"] == "fixed-revision"


def test_manifest_is_written_as_stable_json(tmp_path):
    path = write_run_manifest(config(tmp_path))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert path.name == "run-manifest.json"


def test_config_rejects_invalid_hyperparameters(tmp_path):
    with pytest.raises(ValueError, match="learning_rate"):
        TrainingRunConfig("model", "data", str(tmp_path), learning_rate=0)
