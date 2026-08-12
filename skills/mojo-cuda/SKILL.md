---
name: mojo-cuda
description: >
  Specialty Mojo + CUDA + MLIR coding profile for Grok Build. Use when writing
  GPU kernels, Mojo device code, CUDA twins, MAX structured kernels, MLIR
  progressive lowering, evidence-gated optimization, hypothesis experiments,
  AI-visible structural pattern review, memory layout / tiling, softmax /
  matmul / attention ops, or CUDA→Mojo translation. Triggers on "mojo", "cuda",
  "gpu kernel", "mlir", "shared memory", "tensor core", "mma", "warp",
  "fusion", "bufferize", "evidence", "hypothesis", "active review".
metadata:
  short-description: "Mojo + CUDA + MLIR specialty profile with disk evidence"
---

# Mojo + CUDA Coding Profile v2

Specialty GPU systems profile: Mojo first, CUDA when required, **MLIR-aware
system design**, **active review with disk evidence**, and **hypothesis-driven
experimental implementations**.

## Method (required for structural work)

1. **Altitude** — host API / schedule / high MLIR / GPU dialect / silicon.
2. **Review** — run AI-visible pattern analysis; persist `evidence/review/<id>/`.
3. **Hypothesis** — if the lever is an MLIR capability, state claim + falsify-if;
   implement minimal experiment; persist `evidence/experiment/<id>/`.
4. **Merge only if supported** — cite evidence ids in the answer/PR.

## AI-visible patterns (not single-line lints)

- Latent fusion windows (pure producer→consumer edges)
- Precision drift chains (cast/exp multi-hop)
- Affine nest isomorphism / cross-kernel schedule clones
- Barrier under divergence
- Layout polymorphism collapse
- Multi-scale tile incongruence
- Dialect impedance (early pointer decay)
- Ghost alias surfaces

## System design altitudes

| Layer | Focus |
| --- | --- |
| L0 Product API | shapes, dtypes, ownership |
| L1 Graph schedule | fusion, live ranges |
| L2 High MLIR | linalg/tensor/affine, layouts, effects |
| L3 GPU/NVVM/LLVM | launch, aspace, MMA |
| L4 Silicon | roofline, occupancy, HBM |

**Progressive lowering:** tile/fuse before bufferize; layout is API; side effects honest.

## Disk evidence contract

```
evidence/
  INDEX.jsonl
  review/<id>/evidence.json
  review/<id>/REPORT.md
  experiment/<id>/evidence.json
  experiment/<id>/REPORT.md
  experiment/<id>/impl/*.mojo.txt
```

Never claim a structural optimization is done without an evidence id when this
profile's review tooling is available.

## Output format

1. Plan (bottleneck + altitude)
2. Complete source
3. Host / launch wiring
4. Verification
5. `review_id` / `experiment_id`
6. Risks / next knobs

## Depth references

- `references/system-prompt.md` — full agent system prompt (v2)
- `references/kernel-patterns.md` — patterns & anti-patterns
- `references/mlir-evidence.md` — evidence + hypothesis protocol
- `references/host-runtime.md` — optional host notes (container runtime, local SM)

## Role pack (agents)

For multi-track GPU residual work, spawn specialized roles rather than one
generic agent owning ground + implement + review + optimize:

| Agent | When |
|-------|------|
| `mojo-cuda-implementer` | Write kernels, rebuild, TDD |
| `mojo-cuda-reviewer` | AI-visible + fail-closed review |
| `mojo-cuda-entry-stack` | Dual-entry / Candle attach honesty when that stack is in scope |

Optional host persona overlays may extend these on a given machine; they must
not invent train/serve green or rewrite silicon claims.

## V3 host law (Mojo 1.0 / MAX 26.5 · Ampere sm_86)

- Prefer **`max.gpu.*`** over `std.gpu` as primary for new kernels.
- Fixed-width types · DevicePassable · Pointer honesty · comptime where required.
- Install/activate: orch **`orch-max-mojo-pathway`** (Pixi `max-all` + nvidia + kernel env).
- Product wall authority remains CUDA twins / `run_ampere_3060_stack` until max.gpu leaf is measured.

### V3 references (this package)

- `references/MOJO_CUDA_SPECIALIST_V3_HOST.md`
- `references/MOJO-1-0-MAX-26-5-PROFILE-RESEARCH.md`
- `references/AMPERE-SM86-MOJO-1-0-IMPORT-MIGRATION.md`
- `references/MOJO-CUDA-V3-HUB-POINTER.md`

**Seals:** train_ok=false · measured_omega=false · G1=OPEN · profile ≠ residual science.
