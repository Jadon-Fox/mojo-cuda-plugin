---
name: mojo-cuda
description: >
  Specialty Mojo + CUDA + MLIR coding profile for Grok Build (v3, Mojo 1.0 /
  MAX 26.5). Use when writing GPU kernels, Mojo device code, CUDA twins, MAX
  structured kernels, MLIR progressive lowering, evidence-gated optimization,
  hypothesis experiments, AI-visible structural pattern review, memory layout /
  tiling, softmax / matmul / attention / RMSNorm ops, CUDA→Mojo translation, or
  a GPU host pilot. Triggers on "mojo", "cuda", "gpu kernel", "mlir",
  "shared memory", "tensor core", "mma", "warp", "fusion", "bufferize",
  "evidence", "hypothesis", "active review", "max.gpu", "sm_86", "gpu pilot".
metadata:
  short-description: "Mojo 1.0 + CUDA + MLIR specialty profile (max.gpu, Ampere sm_86)"
---

# Mojo + CUDA Coding Profile v3

Specialty GPU systems profile: **Mojo 1.0 first** (`max.gpu.*`), CUDA when
required, **MLIR-aware system design**, **active review with disk evidence**,
and **hypothesis-driven experimental implementations**.

Supersedes the v2 paste. v2 altitude / evidence / pattern law still holds.

## Mojo 1.0 / MAX 26.5 law (new kernels)

| Do | Don't |
|----|-------|
| `from max.gpu.host import DeviceContext` | Treat `std.gpu.host` / `gpu.host` as default for **new** code |
| `from std.gpu import block_dim, block_idx, thread_idx, global_idx` | Invent `max.gpu.id` without checking the installed MAX |
| Fixed-width DevicePassable (`Int32` / `UInt32` / SIMD / `TileTensor`) | Platform `Int`/`UInt` as kernel args |
| `Pointer` + explicit `unsafe_*` | Silent `UnsafeAnyOrigin` widen / default `UnsafePointer` |
| `comptime` / `comptime if` / `comptime for` tiles | New `@parameter` as primary |
| `layout` (`TileTensor` / `LayoutTensor.as_imm()`) | `get_immutable()` long-term |
| `--target-accelerator=sm_86` on product 3060 | Hopper TMA / WGMMA on 3060 path |
| CUDA twin golden before promote | Workbench vector-add as orch progress |

Canonical host launch (Modular intro tutorial, 1.0):

```mojo
from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from layout import TileTensor, row_major
from std.sys import has_accelerator
```

## GPU pilot leaf (product, not vector-add)

**Kernel:** batched RMSNorm — Unsloth-class math, same as CUDA twin:

`y[m,d] = x[m,d] * rsqrt(mean_d(x[m]^2) + eps) * weight[d]`

| Twin | Path |
|------|------|
| CUDA | orch `mojo-kernels/cuda/rms_norm_batch_ffi.cu` → `launch_rms_norm_batch` |
| Mojo 1.0 | orch `mojo-kernels/mojo/rms_norm_max_gpu.mojo` |
| Smoke | `bash mojo-kernels/scripts/smoke_max_gpu_rms.sh` |
| Charge | orch `docs/designs/2026-08-22-GPU-PILOT-MOJO-1-0-RMSNORM.md` |

`MOJO_GPU_PILOT_RMS_OK` = compile + host golden on **this** run. Not train_ok.
Not measured_omega. Not Unsloth-beat. Not residual-body graph.

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
- **Import lag** (`std.gpu.host` / `UnsafePointer` / `@parameter` in new leafs)
- **Hopper leakage** (TMA/WGMMA on sm_86 charges)

## System design altitudes

| Layer | Focus |
| --- | --- |
| L0 Product API | shapes, dtypes, ownership |
| L1 Graph schedule | fusion, live ranges |
| L2 High MLIR / max.gpu | linalg/tensor/affine, layouts, DeviceContext |
| L3 GPU/NVVM/LLVM | launch, aspace, MMA |
| L4 Silicon | roofline, occupancy, HBM · sm_86 12 GiB |

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

- `references/system-prompt.md` — full agent system prompt (v3)
- `references/kernel-patterns.md` — patterns & anti-patterns (incl. max.gpu)
- `references/mlir-evidence.md` — evidence + hypothesis protocol
- `references/host-runtime.md` — optional host notes (2080 sm_75 / 3060 sm_86)
- `references/MOJO_CUDA_SPECIALIST_V3_HOST.md` — host paste SSOT
- `references/MOJO-1-0-MAX-26-5-PROFILE-RESEARCH.md`
- `references/AMPERE-SM86-MOJO-1-0-IMPORT-MIGRATION.md`
- `references/MOJO-CUDA-V3-HUB-POINTER.md`
- `references/GPU-PILOT-LEAF.md` — RMSNorm GPU-pilot contract

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

Install/activate: orch **`orch-max-mojo-pathway`** (Pixi `max-all` + nvidia + kernel env)
when that charge is on disk. Product wall authority remains CUDA twins /
`run_ampere_3060_stack` until a max.gpu leaf is **measured on the GPU host**.

**Seals:** train_ok=false · measured_omega=false · G1=OPEN · profile ≠ residual science.
