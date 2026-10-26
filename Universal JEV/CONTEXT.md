# Universal JEV Context

This file is the project glossary. It contains domain language, not implementation details.

## Decision

The act of selecting one candidate from a finite, application-provided set in response to a context.

## Decision context

The information supplied to Universal JEV that the model uses to evaluate candidates.

## Candidate

An allowed alternative in a decision. A candidate has a stable machine-readable ID and a human-readable description.

## Decision engine

The domain capability that evaluates candidates and produces a decision result. It does not execute external tools.

## Exact scoring

Candidate evaluation using token-level conditional log likelihoods under a model with accessible logits or equivalent token probabilities.

## Structured-generation fallback

Candidate selection by asking a model to emit an allowed candidate ID through a constrained or validated generation interface. This is an approximation and is not paper-equivalent exact scoring.

## Decision result

The auditable output of a decision: selected candidate or abstention, candidate scores, normalized weights when meaningful, policy outcome, and provenance metadata.

## Abstention

A valid outcome in which the engine declines to select a candidate because configured decision-policy conditions are not met.

## Decision policy

The rules applied after scoring to determine selection, abstention, tie handling, and future fallback behavior.

## Normalized score

A distribution derived from comparable candidate scores, such as a softmax over log scores. It is not automatically a calibrated probability of correctness.

## Provider adapter

An integration boundary that exposes a model's decision capabilities and declares which scoring modes it supports.

## Decision execution

The responsibility of the calling application to authorize and perform the action selected by Universal JEV. It is outside the decision engine boundary.
