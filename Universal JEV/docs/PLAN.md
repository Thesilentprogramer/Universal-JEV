# Universal JEV Development Plan

## Phase 0 — Foundations

- Establish the Python 3.11 package with `src/` layout.
- Define decision, candidate, result, scoring, policy, and provider capability contracts.
- Add CI-quality checks with pytest and Ruff.
- Version the paper-inspired prompt template.

## Phase 1 — Exact local MVP

- Implement the Hugging Face adapter for Qwen2.5-1.5B/3B Instruct.
- Implement numbered candidate rendering and the fixed continuation prefix.
- Score complete candidate suffixes, including multi-token suffixes.
- Handle tokenizer and prefix-alignment edge cases.
- Normalize comparable log scores with numerically stable softmax.
- Return model, tokenizer, template, and scoring metadata.
- Add minimum-score and top-two-margin abstention policies.

### MVP acceptance criteria

- A developer can call the SDK with a context and nonempty candidate list.
- Candidate IDs are preserved in the result.
- The scorer evaluates all candidates under the same conditioning context.
- Results are deterministic under fixed model and inference settings.
- Tests cover reordered candidates, multiple candidates, multi-token labels, invalid input, ties, and abstention.
- The tool-selection fixture suite produces inspectable score and policy outputs.

## Phase 2 — Reliability and evaluation

- Compare exact scoring with ordinary free-form generation.
- Add accuracy, latency, cost, Brier score, and ECE evaluation reports.
- Add batch evaluation and exportable result records.
- Test calibration before allowing confidence thresholds to drive consequential automation.

## Phase 3 — Provider expansion

- Add capability discovery.
- Add exact scoring where hosted APIs expose suitable token probabilities.
- Add explicitly labeled structured-generation fallback where they do not.
- Add FastAPI as a thin transport over the SDK.

## Phase 4 — Research fine-tuning

- Add reproducible labeled decision datasets.
- Implement LoRA-based flat listwise training before separately specifying a tree-factorized objective.
- Add KL regularization and compare against the frozen baseline.
- Evaluate in-domain and out-of-domain behavior, calibration, and general-generation preservation.

## Phase 5 — Ecosystem integrations

- Add LangGraph and other agent-framework integrations.
- Add optional evaluation dashboard.
- Document operational safety boundaries and deployment patterns.
