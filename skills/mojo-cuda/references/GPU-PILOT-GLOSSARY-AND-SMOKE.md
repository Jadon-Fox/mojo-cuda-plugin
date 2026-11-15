# GPU-pilot glossary + smoke ladder (LCD)

**Stamp:** 2026-08-22  
**Skills:** lcd-glossary-integrity · honest-prompt-rewrite · twinglass-lattice (scout; **fork unearned**)  

## Locked terms

| Term | Referent (one) |
|------|----------------|
| **GPU_PILOT_LEAF** | First **product-shaped** Mojo 1.0 `max.gpu` kernel that **compiles on host GPU** and matches a **CPU golden**. Token: `MOJO_GPU_PILOT_RMS_OK`. |
| **PRODUCT_RMSNORM_TWIN** | Algorithm identity with `launch_rms_norm_batch`: **one block per row**, 256-thread **shared** tree reduce, single global pass over `x`. Token: `MOJO_GPU_PILOT_RMS_TWIN_OK` — **not emitted yet**. |
| **H-RMS-SMEM** | Hypothesis that a fused Mojo kernel **is** PRODUCT_RMSNORM_TWIN. Separate file. Optional `ORCH_RMS_FUSE=1`. |
| **compile_golden** | `claim_class` for GPU_PILOT_LEAF this-run. Not wall. Not twin. |
| **UNMEASURED** | Host has not produced the token. This sandbox cannot. |
| **sm_86 product** | RTX 3060 12 GiB **GA106**. See fact card. |

**Empty meet (obtuse, killed):** using one name “the RMSNorm leaf” for both GPU_PILOT_LEAF and PRODUCT_RMSNORM_TWIN.

Lattice: after this lock, two-pass vs fused are **specializations of different terms**, not exclusive accounts under one lock. **No fork.**

## Smoke expectations (`smoke_max_gpu_rms.sh`)

| Exit | Stdout token / reason | `claim_class` | What it proves | What it does **not** |
|------|------------------------|---------------|----------------|----------------------|
| 127 | `mojo not on PATH` | n/a | env missing | anything GPU |
| ≠0, `NO_ACCELERATOR` | no device | `UNMEASURED` | process saw no GPU | compile of kernels |
| ≠0, `golden_fail` | `max_abs` > 1e-5 | `UNMEASURED` | compiled but numbers wrong | twin / wall |
| **0** | **`MOJO_GPU_PILOT_RMS_OK`** | **`compile_golden`** | this-run compile + CPU golden, M=4 D=3072, atol 1e-5 | twin, wall, occupancy, training_verified, Ω |
| (optional fuse, later) | `MOJO_GPU_PILOT_RMS_TWIN_OK` | `cuda_twin` | fused schedule + same buffers as CUDA | training_verified |

JSON schema `mojo_gpu_pilot_v1` on every path (ok or fail). `training_verified` always `false`. `measured_viability` always `false`. `primary_gate` always `OPEN`. `twin_compared` stays `false` until CUDA buffers are joined.

### Pass line you should see on 40-Runtime-Deps

```
ENTRY_MOJO_GPU_PILOT kernel=rms_norm_max_gpu M=4 D=3072 max_abs=<1e-5> atol=1e-5
MOJO_GPU_PILOT_RMS_OK
{"schema":"mojo_gpu_pilot_v1",...,"claim_class":"compile_golden","pass":"two","twin_compared":false,...}
```

`max_abs` on CUDA unit smoke was ~1.2e-7 (fp32). 1e-5 is the **pilot gate**, not the twin bar.

### Fail lines (honest)

```
NO_ACCELERATOR          → this machine / container has no GPU
import max.gpu.host     → Mojo < 1.0 or hall b2 std mixed with modular 1.0
golden_fail             → math or layout bug; do not stamp OK
```

## Morph contract (B-primary research — plane honesty)

```
TELOS: compile+golden a decoder RMSNorm max.gpu leaf so the 3060 can pilot Mojo 1.0
MARKERS:
  M-SILENT-RENAME-LEAF  GPU_PILOT_LEAF ≠ PRODUCT_RMSNORM_TWIN
  M-GA106-NOT-GA102     3060 die is GA106
  M-RMS-NOT-MMA         this stage is not Tensor Core
  M-M4-UNDERSUB         M=4 is golden, not 28-SM occupancy
  M-PLANE-SUBSTRATE     kernel wall ≠ Mythos ≠ Ω
WARRANTED_EVIDENCE: silicon table (fact card); CUDA twin source; Modular DeviceContext + barrier docs
OPEN: CUDA twin join; H-RMS-SMEM measure; LLMVE residual co-emission (other plane)
NON_CLAIMS: training_verified measured_viability primary_gate endpointAssumed  category_grain
```

Hypothesis A sensors are **not** this work. Do not promote RMSNorm golden to Mythos.
