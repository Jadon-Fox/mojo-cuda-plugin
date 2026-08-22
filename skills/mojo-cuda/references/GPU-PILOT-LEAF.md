# GPU pilot leaf — batched RMSNorm (Mojo 1.0)

**Stamp:** 2026-08-22
**Hypothesis:** H-MAX-GPU-REHOME + H-CUDA-TWIN on a **product** op (not vector-add).
**Seals:** train_ok=false · measured_omega=false · G1=OPEN · UNMEASURED until host smoke.

## Why RMSNorm

Every decoder layer runs it. CUDA twin already exists and has host goldens
(`max_abs ≈ 1.2e-7` on sm_75, 2026-07-29). A max.gpu port is a real dual-entry
leaf, not a workbench toy.

## Math (locked with CUDA twin)

```
y[m, d] = x[m, d] * rsqrt(mean_d(x[m]^2) + eps) * weight[d]
layout: row-major [M, D]
weight NULL / ones → scale 1
eps: 1e-5 (phi4_mini) · 1e-6 (r1)
```

CUDA: one block per row, 256-thread shared reduction.
Mojo 1.0 pilot: **two-pass** (row-reduce then apply) using only documented
`DeviceContext` + `TileTensor` APIs. Shared-memory fused pass is a later
hypothesis (H-RMS-SMEM), not this leaf.

## Paths (orch tree)

| Piece | Path |
|-------|------|
| Mojo | `mojo-kernels/mojo/rms_norm_max_gpu.mojo` |
| Smoke | `mojo-kernels/scripts/smoke_max_gpu_rms.sh` |
| CUDA twin | `mojo-kernels/cuda/rms_norm_batch_ffi.cu` → `launch_rms_norm_batch` |
| Charge | `docs/designs/2026-08-22-GPU-PILOT-MOJO-1-0-RMSNORM.md` |

## Success / abort

| | |
|--|--|
| **Success** | `mojo run` on GPU host · `MOJO_GPU_PILOT_RMS_OK` · CPU golden max_abs < 1e-5 · JSON board `claim_class=measured_compile` still **not** train_ok |
| **Abort** | no accelerator · import fail (`std.gpu.host` vs `max.gpu.host`) · golden fail · OOM |
| **Non-claims** | train_ok · measured_omega · Unsloth-beat · residual-body graph · ENTRY_C replace |

## Next after this leaf measures

1. Fuse two-pass → shared reduction (CUDA-congruent).
2. Phi-4 D=3072 + R1 D=1536 product-shape goldens vs `launch_rms_norm_batch`.
3. Optional H-TILE / NF4 leaf under the same import law.
