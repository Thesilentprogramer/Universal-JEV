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
