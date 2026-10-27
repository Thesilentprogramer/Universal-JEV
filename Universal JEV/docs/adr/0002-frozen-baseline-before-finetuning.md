# ADR 0002: Establish a frozen baseline before fine-tuning

## Status

Accepted

## Context

The project may eventually implement the paper's listwise/tree-factorized objective with KL regularization and LoRA. Adding training before measuring the frozen model would make it difficult to attribute improvements and would increase project scope prematurely.

## Decision

Fine-tuning follows the frozen-model MVP and its initial evaluation. The repository may contain a training scaffold (data contract, losses, and optional LoRA preparation), but no model should be trained until the frozen baseline is measured and a reproducible experiment configuration is approved.

## Consequences

- Baseline correctness and calibration are measured first.
- Fine-tuning work has a concrete comparison target.
- The training scaffold is not evidence that fine-tuning is complete or beneficial.
- Any training run must compare against the frozen baseline and record model, data, template, and loss configuration.
