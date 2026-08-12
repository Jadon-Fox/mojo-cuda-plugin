# Ampere sm_86 — Mojo 1.0 / MAX 26.5 import migration

**Stamp:** 2026-08-11  
**Target:** RTX 3060 · sm_86 · product path  
**Pairs with:** Mojo+CUDA Specialist **v3** · Ampere methodology  
**Seals:** `train_ok=false` · `measured_omega=false`  

---

## Package rehome (new code)

| Legacy (pre-1.0 teaching) | Mojo 1.0 / MAX 26.5 |
|---------------------------|---------------------|
| `std.gpu.host` / `gpu.host` | `max.gpu.host` |
| `std.gpu.compute` | `max.gpu.compute` |
| `std.gpu.memory` | `max.gpu.memory` |
| `std.gpu.sync` | `max.gpu.sync` |
| `std.algorithm` (accel) | `max.algorithm` |
| `std.benchmark` multicontext | `max.benchmark` |
| layout as pure stdlib | layout **bundled with MAX** |
| `LayoutTensor.get_immutable()` | `LayoutTensor.as_imm()` (deprecated old name) |

Doc snippets may lag — **changelog is authority for new kernels**.

---

## Type / safety migration

```text
DevicePassable:   Int32 / UInt32 / Float32 / SIMD[...]   — NOT platform Int/UInt
Pointers:         Pointer  +  unsafe_load / unsafe_store
Origins:          no silent UnsafeAnyOrigin widen
Decls:            var required
Metaprogram:      comptime / comptime if / comptime for
Destructors:      __deinit__ (not __del__)
SIMD:             .length (not .size)
InlineArray:      Array[..., length=...]
StringSlice:      StringSpan
Negative index:   forbidden on collections
```

---

## Product sm_86 checklist

```text
[ ] mojo build --target-accelerator=sm_86  (or orch build wrapper)
[ ] No TMA / WGMMA / sm_90-only
[ ] Shared tile ≤ L1/shm budget after occupancy
[ ] TF32 policy: wall vs science explicit
[ ] CUDA twin golden named
[ ] ONE_PRODUCT A+B
[ ] review_id / experiment_id if structural
```

---

## Minimal host pattern (illustrative)

```mojo
# Illustrative only — verify against installed max.gpu API on host
from max.gpu.host import DeviceContext
# from max.gpu.id import block_dim, block_idx, thread_idx  # if re-exported here
# from layout import Layout, LayoutTensor

fn kernel_example(...):
    ...

def main():
    var ctx = DeviceContext()
    # enqueue / load_function with fixed-width params
    ...
```

Prefer MAX custom-op / DeviceContext patterns used in Modular examples after 1.0 rehome.

---

## Non-claims

Migration doc ≠ live kernel ship · ≠ train_ok · ≠ measured_omega · ≠ Unsloth-beat.
