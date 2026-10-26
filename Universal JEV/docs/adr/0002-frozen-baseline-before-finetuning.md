# ADR 0002: Establish a frozen baseline before fine-tuning

## Status

Accepted

## Context

The project may eventually implement the paper's listwise/tree-factorized objective with KL regularization and LoRA. Adding training before measuring the frozen model would make it difficult to attribute improvements and would increase project scope prematurely.

## Decision

Fine-tuning is a later research phase. The MVP and initial evaluation use frozen Qwen weights, exact scoring, and deterministic tool-selection fixtures.

## Consequences

- Baseline correctness and calibration are measured first.
- Fine-tuning work has a concrete comparison target.
- Training code is intentionally absent from the MVP rather than partially implemented.
