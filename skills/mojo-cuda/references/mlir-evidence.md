# MLIR evidence & hypothesis protocol

## Why disk evidence

Structural GPU opts fail silently when justified by intuition. This profile requires
durable artifacts so agents and humans can audit:

- which AI-visible patterns fired
- which hypothesis was tested
- before/after IR metrics
- experimental source that claimed support

## Paths

```
evidence/INDEX.jsonl
evidence/review/<id>/evidence.json
evidence/review/<id>/REPORT.md
evidence/experiment/<id>/evidence.json
evidence/experiment/<id>/REPORT.md
evidence/experiment/<id>/impl/<hypothesis>.mojo.txt
```

## Review payload (minimum)

- `findings[]`: patternId, severity, score, humanBlindSpot, mlirAngle, recommendation
- `irs[]`: lightweight node/edge metrics per source
- `summary`: total / critical / watch / avgScore

## Experiment payload (minimum)

- hypothesis: claim, mlirCapability, falsifyIf
- before/after metrics + delta
- status: supported | falsified | inconclusive
- experimentalSource

## Decision rule

| Status | Action |
| --- | --- |
| supported | May merge; cite experiment id + residual risks |
| falsified | Do not merge; document counter-evidence |
| inconclusive | Need better metric or larger experiment |

## MLIR capabilities commonly under test

- Late bufferization preserving fusion
- Transform-dialect shared tile lattices
- MemoryEffects-honest pure regions
- Typed layout encodings before pointer decay
- Barrier legality on structured CFG
