# Kernel patterns & anti-patterns (v3)

## Prefer

- Product leaf first: **batched RMSNorm** (one block/row CUDA twin, or two-pass max.gpu)
- 1D elementwise with coalesced `base + tid` addressing as a *compile* smoke — not orch progress
- Tiled matmul with shared tiles, barriers, and tail masks
- Online softmax for attention rows (stable max + sum)
- Structured pipelines for production GEMM / fused epilogues
- Host launch sizing: `blocks = ceildiv(n, threads)` via `std.math.ceildiv`
- `DeviceContext.enqueue_function[k]` + host/device buffers + `enqueue_copy` + `synchronize`

## Mojo 1.0 launch skeleton

```mojo
from max.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx
from std.math import ceildiv
from layout import TileTensor, row_major
from std.sys import has_accelerator
```

## Anti-patterns

| Avoid | Prefer |
| --- | --- |
| Unmasked stores past N | `if i < n` / epilogue masks |
| Shared write then read without barrier | `barrier()` between phases |
| Column-strided global loads in warps | Transpose staging or layout change |
| One giant kernel file | Config + pipelines + thin launch |
| Claiming “optimal” without roofline | Name bottleneck + expected gain class |
| Invented Modular imports | Sketches marked with assumptions |
| `std.gpu.host` / `gpu.host` in **new** kernels | `max.gpu.host.DeviceContext` |
| Platform `Int` as DevicePassable | `Int32` / `TileTensor` / SIMD |
| Hopper TMA/WGMMA on sm_86 | Ampere-legal async / classic GEMM |
| Vector-add as ENTRY_MOJO progress | RMSNorm / RoPE / residual / SDPA leaf |
| Compile OK ⇒ training_verified | Golden + claim_class=UNMEASURED |

## Review order

1. Correctness / shapes
2. Import law (`max.gpu.host`, `std.gpu` ids)
3. Global memory behavior
4. Shared memory reuse & banks (or named two-pass)
5. Register pressure
6. Algorithmic fusion / TC shapes
7. CUDA twin golden named
8. gates honest
