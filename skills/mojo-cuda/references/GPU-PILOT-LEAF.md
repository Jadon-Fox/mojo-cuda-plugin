# GPU pilot leaf — batched RMSNorm (Mojo 1.0)

**Stamp:** 2026-08-22 (glossary split)  
**Hypothesis:** H-MAX-GPU-REHOME on a **product stage** (not vector-add).  
**Seals:** train_ok=false · measured_omega=false · G1=OPEN · UNMEASURED until host smoke.

## LCD (do not silent-rename)

| Term | Token | Proves |
|------|-------|--------|
| **GPU_PILOT_LEAF** | `MOJO_GPU_PILOT_RMS_OK` | two-pass `max.gpu` compile + CPU golden |
| **PRODUCT_RMSNORM_TWIN** | `MOJO_GPU_PILOT_RMS_TWIN_OK` | fused 1-block/row (optional `ORCH_RMS_FUSE=1`) — still not CUDA-buffer join |
| **H-CUDA-TWIN** | not emitted | same buffers as `launch_rms_norm_batch` |

`claim_class=compile_golden` on OK. Never `train_ok`.

## Why RMSNorm

Decoder residual-stream stage (pre-attn, pre-MLP, final) — **not** a matmul, **not** Tensor Core. CUDA twin already has host goldens (`max_abs ≈ 1.2e-7` on sm_75, 2026-07-29).

## Math

```
y[m, d] = x[m, d] * rsqrt(mean_d(x[m]^2) + eps) * weight[d]
layout: row-major [M, D] · weight ones · eps 1e-5 · smoke M=4 D=3072
```

M=4 is **golden**, not 28-SM occupancy. Product silicon is **GA106 sm_86**, not GA102.

## Paths (orch tree)

| Piece | Path |
|-------|------|
| Two-pass leaf | `mojo-kernels/mojo/rms_norm_max_gpu.mojo` |
| Fused candidate | `mojo-kernels/mojo/rms_norm_max_gpu_smem.mojo` |
| Smoke | `mojo-kernels/scripts/smoke_max_gpu_rms.sh` |
| CUDA twin | `mojo-kernels/cuda/rms_norm_batch_ffi.cu` → `launch_rms_norm_batch` |
| Charge | `docs/designs/2026-08-22-GPU-PILOT-MOJO-1-0-RMSNORM.md` |
| Smoke ladder | `docs/designs/2026-08-22-GPU-PILOT-GLOSSARY-AND-SMOKE.md` |

## Success / abort

| | |
|--|--|
| **Success** | `MOJO_GPU_PILOT_RMS_OK` · max_abs < 1e-5 · JSON `claim_class=compile_golden` |
| **Abort** | no accelerator · import fail · golden fail · OOM · Hopper APIs on sm_86 |
| **Non-claims** | train_ok · measured_omega · Unsloth-beat · wall · occupancy · Ω · Mythos |

Fused APIs (documented): `from max.gpu.sync import barrier` + `stack_allocation[..., address_space=AddressSpace.SHARED]`. Not TMA / SharedMemBarrier.
