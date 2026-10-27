# Universal JEV

Universal JEV is an open-source decision inference layer for using large language models as explicit, auditable decision engines.

The project begins with a research-faithful implementation of candidate likelihood scoring on local Qwen2.5 Instruct models. It will later add hosted-provider fallbacks, evaluation tooling, and paper-inspired fine-tuning.

## Current focus

The MVP is a Python SDK for agent tool selection:

1. accept a context and stable candidate IDs/descriptions;
2. render one versioned numbered-choice prompt;
3. score candidate suffixes using token-level log likelihoods;
4. normalize comparable scores;
5. apply abstention policy;
6. return a decision and transparent metadata.

The first provider is a local Hugging Face adapter for Qwen2.5-1.5B/3B Instruct. Hosted providers, FastAPI, broader benchmarks, and fine-tuning are later phases.

## Training scaffold

The optional `training` extra provides LoRA preparation and composable losses for
supervised decision fine-tuning. Training examples use JSON Lines with a context,
candidates, and a labeled `target_id`; see `examples/tool-selection.jsonl`.

```bash
uv sync --extra training
```

`combined_loss` combines listwise decision supervision with optional
KL(reference || current) preservation. It is a foundation for the paper-inspired
training phase, not a claim that the complete training pipeline or its results
have already been reproduced.

### Local listwise training

The initial runner performs one labeled decision per optimizer step and is offline
by default. Model weights must already exist locally or in the Hugging Face cache:

```bash
universal-jev-train \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data examples/tool-selection.jsonl \
  --output runs/qwen-tool-selection \
  --revision <pinned-revision>
```

Pass `--allow-downloads` only when network model retrieval is intentional. The
runner writes `run-manifest.json` before optimization and saves adapters through
the model's `save_pretrained` interface. KL-preservation training is not wired
into this first runner; nonzero KL configuration is rejected rather than ignored.

## Repository map

- `CONTEXT.md` — project glossary and domain language
- `docs/PLAN.md` — phased development plan and acceptance criteria
- `docs/adr/` — durable architectural decisions
- `src/universal_jev/` — implementation, once development begins
- `tests/` — deterministic scorer and policy tests

## Design principles

- Exact scoring and approximate generation are never presented as equivalent.
- Normalized scores are not called calibrated probabilities without empirical validation.
- The decision layer recommends an action; a separate application policy authorizes execution.
- Reproducibility requires recording model, tokenizer, template, and scoring metadata.
