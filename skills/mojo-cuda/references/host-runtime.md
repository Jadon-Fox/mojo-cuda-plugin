# Optional host runtime notes (Ms-2080 example)

Not required for the portable profile. Use when this machine matches.

# Local host overlay — Ms-2080 (not a profile rewrite)

| Field | Value |
|-------|--------|
| GPU | RTX 2080 · **sm_75** (not SM 8.0 default for local green) |
| Runtime | LXD `40-Runtime-Deps` · `pixi` / `mojo` / `max` |
| Product kernels | `/home/workspace/training_orchestrator/mojo-kernels/` |
| Shared CUDA ABI | `…/artifacts/build_20260719/lib/*.so` |
| ENTRY_C SSOT | `model_step_real` (weights / full step) |
| ENTRY_MOJO | inject + product layer smoke dual-entry |
| Evidence root (local) | `mojo-kernels/evidence/{review,experiment}/` |
| Seals | train_ok=false · measured_omega=false · G1=OPEN · no Triton/Python progress |

SM 8.0+ in the upstream profile remains a **portability / Modular target note**, not local occupancy claims.

## Personas v2

Mandatory overlay: `~/.grok/personas/_host_overlay_mojo_cuda.md`  
Role table: `~/.grok/personas/README_MOJO_CUDA.md`
