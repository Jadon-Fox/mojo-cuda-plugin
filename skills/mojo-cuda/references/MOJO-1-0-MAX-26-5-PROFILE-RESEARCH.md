# Mojo 1.0 + MAX 26.5 — Mojo+CUDA profile research digest

**Stamp:** 2026-08-11  
**Method:** morph-shared shared mode (expand → disconfirm → CONTRACT_LRR) + external harvest  
**Applies to:** `docs/designs/agent_profiles/MOJO_CUDA_SPECIALIST_V3_HOST.md`  
**Seals:** `train_ok=false` · `measured_omega=false` · `G1=OPEN` · profile ≠ product residual science  

---

## Telos

Upgrade the Mojo+CUDA specialist profile with **content beyond early Modular agent skills** — specifically Mojo **1.0 stable** (Modular platform **26.5**, 2026-08-11), GPU package rehome, Ampere sm_86 coding rules, and honest multi-vendor caveats.

---

## Sources (warranted)

| Claim | Necessary because | Source |
|-------|-------------------|--------|
| Mojo 1.0.0 ships with stability policy + GPU package move | Profile import paths must not teach dead `std.gpu` as primary | [mojolang.org/releases/v1.0.0](https://mojolang.org/releases/v1.0.0/) · [docs.modular.com/releases/v26.5](https://docs.modular.com/releases/v26.5/) |
| `std.gpu.*` → `max.gpu.*`; layout bundled with MAX | Kernel templates break without rehome | MAX 26.5 changelog |
| `Int`/`UInt` not DevicePassable — use fixed-width | Silent miscompile risk on GPU kernels | Mojo 1.0 + MAX 26.5 |
| Pointer unification (`Pointer` + `unsafe_*`) | Stop teaching `UnsafePointer` as default | Mojo 1.0 changelog |
| `comptime` replaces `@parameter` for compile-time | Agent-generated kernels need current metaprogramming | Modular 26.2 language notes |
| Modular skills: kernel translate + model serve/bringup | Agents should install upstream skills, not reinvent | [github.com/modular/skills](https://github.com/modular/skills) · Modular 26.2 / 26.4 |
| Multi-vendor MAX (CUDA/HIP/Metal) from one Mojo codebase | Portability is real; silicon still constrains MMA | Modular platform + docs |
| Ampere 3060 = sm_86: no Hopper TMA/WGMMA | Host target remains consumer GA | Product Ampere util locks |
| ORNL SC25: Mojo competitive memory-bound vs CUDA/HIP | Evidence for Mojo-first, not CUDA-only | arXiv:2509.21039 |
| ModCon 2026 / open-source compiler path | Roadmap honesty — not invent OSS-complete | Modular blog / Chris Lattner posts |

---

## Pattern markers (ration bank)

| ID | Marker |
|----|--------|
| M-IMPORT-MAX-GPU | Prefer `from max.gpu.host import DeviceContext` (and `max.gpu.*`); treat bare `std.gpu` as legacy |
| M-FIXED-WIDTH | Kernel params use `Int32`/`UInt32`/SIMD dtypes — never platform `Int` as DevicePassable |
| M-POINTER-SAFE | `Pointer` + explicit `unsafe_*`; no implicit `UnsafeAnyOrigin` |
| M-COMPTIME | Tile lattices via `comptime` / `comptime if` / `comptime for` |
| M-LAYOUT-TENSOR | `LayoutTensor` + layout package (MAX-bundled) before raw aspace soup |
| M-SPAN-ASPACE | `Span[..., address_space=...]` for shared/shared-like surfaces |
| M-CUDA-TWIN | CUDA/host golden before Mojo promote; KD-16 no raw Lt handles in Mojo |
| M-SM86-NO-HOPPER | No TMA / WGMMA / sm_90-only paths on 3060 product |
| M-GRAPH-STABLE | Capture only frozen topology; recapture on shape/inject/dumps |
| M-AGENT-SKILLS | Install `modular/skills` for CUDA→Mojo + serve/import/debug-model |
| M-EVIDENCE-GATE | review_id + experiment_id before structural “done” |
| M-THREE-BOOKS | Wall ms / FLOPS ≠ Φ_E; never invent train_ok / measured_omega |

---

## Beyond “recent Modular abilities” (net-new for profile)

Content that is **not** just “skills exist”:

1. **Mojo 1.0 stability surface** — which traits/types are stable (`Copyable`, `Movable`, `List`, `Span`, `Array`, …) and which GPU APIs moved into MAX.
2. **Import / package rehome map** — concrete `std.*` → `max.*` table for kernels, benchmarks, layout.
3. **DevicePassable rules** — fixed-width types, `Coord` passability, address-space extensibility (`SCRATCHPAD`, etc.).
4. **Warp / MMA expansions** — `warp.match_any/all`, Apple simdgroup 8×8, SM100 FP8 MLA (portability vs product sm_86).
5. **DeviceGraphBuilder / external PTX-SASS** — compile-time kernel params + load_function cache keys (entry + blob hash).
6. **Serving-adjacent kernels** — grammar bitmask fused op, int8 KV, LoRALinear / StackedLinearLoRA / fused QKV (context for orch LoRA, not invent train_ok).
7. **Agent skill ladder** — mojo-gpu-fundamentals, CUDA/Triton→Mojo translation, import-model / debug-model, serve/benchmark/eval/profile-model.
8. **Hardware matrix honesty** — B300/Jetson Thor/DGX Spark/RDNA consumer/MI355X/M1–M5 listed upstream; **product host remains RTX 3060 sm_86 12 GiB**.
9. **Language hygiene for agents** — `var` required, no negative index, `SIMD.length`, `InlineArray`→`Array`, `__deinit__`, lambda closures, interior origins on buffers.
10. **Research-grade evidence** — portable Mojo vs CUDA/HIP study; do not claim universal CUDA beat.

---

## Disconfirm (what we refuse)

| Tempting claim | Why refuse |
|----------------|------------|
| “Mojo 1.0 = product train_ok” | Language release ≠ orch residual science |
| “max.gpu = drop-in CUDA graphs full step” | Capture topology still product-owned; R7/fidelity bar independent |
| “Multi-vendor means one binary wins all silicon” | MMA/async primitives differ; measure per target |
| “std.gpu still canonical in 1.0 docs snippets” | Changelog rehome is authority for new code; dual-doc lag expected |
| “Agent skills alone close H-TILE / fuse” | Skills help translate; goldens + wall packs still required |

---

## Migration checklist (kernel code)

```text
[ ] Imports: max.gpu.host / max.gpu.compute / max.gpu.memory / max.gpu.sync
[ ] Layout: MAX-bundled layout / LayoutTensor; LayoutTensor.as_imm() not get_immutable()
[ ] Types: fixed-width DevicePassable args
[ ] Pointers: Pointer + unsafe_* only where needed
[ ] Metaprogramming: comptime not @parameter for new code
[ ] Target: --target-accelerator=sm_86 for product; never sm_90-only intrinsics
[ ] Goldens: CUDA twin or host ref before promote
[ ] Evidence: review_id + experiment_id for structural opts
[ ] Seals: train_ok=false measured_omega=false
```

---

## Open frontiers

1. Live dual-check: host container Mojo version string vs 1.0.0 changelog (runtime placement).
2. Promote one product NF4/H-TILE leaf under max.gpu imports with goldens.
3. Optional: wire modular/skills install into host CLI competence charge (not invent green).
4. Keep CUDA twin for cuBLASLt / AdaLomo / Muon FFI until Mojo ports measured.

---

## Non-claims

Research digest ≠ train_ok · ≠ measured_omega · ≠ Unsloth-beat · ≠ full residual-body graph default · ≠ energy/exergy identity from wall.

**Morph seals held.**
