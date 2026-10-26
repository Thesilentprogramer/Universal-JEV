# ADR 0001: Make exact local scoring the reference implementation

## Status

Accepted

## Context

The paper-inspired method depends on candidate token likelihoods. Hosted LLM APIs do not expose equivalent probability information consistently, while local causal models can expose logits. A universal interface is feasible, but a universal exact scoring implementation is not guaranteed.

## Decision

Universal JEV will first implement exact candidate likelihood scoring on a local Qwen2.5 Instruct model. Hosted-provider support and structured-generation fallbacks will be added only after the reference path is validated.

## Consequences

- The MVP has a measurable research baseline.
- Approximate provider behavior cannot silently masquerade as exact paper reproduction.
- Local model dependencies are heavier, so they remain optional for core installation.
- The provider interface must declare scoring capabilities explicitly.
