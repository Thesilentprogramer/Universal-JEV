"""Command-line entrypoint for explicit local LoRA training runs."""

import argparse

from .experiment import TrainingRunConfig
from .runner import run_training
from .training import load_jsonl, prepare_lora_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Universal JEV on labeled decisions")
    parser.add_argument("--model", required=True, help="Local path or Hugging Face model ID")
    parser.add_argument("--data", required=True, help="Decision training JSONL path")
    parser.add_argument("--output", required=True, help="Directory for the manifest and adapter")
    parser.add_argument("--revision", help="Pinned model revision")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow Transformers to fetch uncached model files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional environment
        raise RuntimeError("the training command requires the 'training' extra") from exc

    torch.manual_seed(args.seed)
    load_options = {
        "revision": args.revision,
        "local_files_only": not args.allow_downloads,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_options)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, **load_options)
    model = prepare_lora_model(base_model)
    config = TrainingRunConfig(
        base_model=args.model,
        model_revision=args.revision,
        dataset_path=args.data,
        output_dir=args.output,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        kl_weight=0.0,
    )
    summary = run_training(model, tokenizer, load_jsonl(args.data), config)
    print(
        f"completed {summary.step_count} steps; mean loss={summary.mean_loss:.6f}; "
        f"manifest={summary.manifest_path}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
