# Optional host runtime notes

Not required for the portable profile. Use when this machine matches.

## Product silicon (Ampere — GPU pilot target)

| Field | Value |
|-------|--------|
| GPU | RTX **3060** · **sm_86** · 12 GiB |
| Runtime | LXD `40-Runtime-Deps` · pixi modular / Mojo **1.0** / MAX **26.5** |
| MODULAR_HOME | `/root/.pixi/envs/modular/share/max` (do not mix hall b2 std with 1.0 compiler) |
| Product kernels | `/home/workspace/training_orchestrator/mojo-kernels/` |
| GPU pilot | `mojo/rms_norm_max_gpu.mojo` · `scripts/smoke_max_gpu_rms.sh` |
| Shared CUDA ABI | `…/artifacts/build_20260719/lib/*.so` |
| ENTRY_C SSOT | `model_step_real` (weights / full step) |
| ENTRY_MOJO | inject + product layer smoke dual-entry (CUDA `.so`, still 1.0b3 FFI until re-measured) |
| gates |  · no Triton/Python progress |

## Dual host (Turing — historical ENTRY_MOJO)

| Field | Value |
|-------|--------|
| GPU | RTX **2080** · **sm_75** (not SM 8.0 occupancy claims) |
| Runtime | same `40-Runtime-Deps` container when that box is live |
| Evidence | `mojo-kernels/evidence/{review,experiment}/` · `SMOKE_20260729.md` |

SM 8.0+ in upstream Modular notes is a **portability / MAX target**, not a local occupancy claim.

Do not mix `mojo 1.0.0b2` hall std with `1.0` / `26.5` compiler (`MOJO_ENV_NOTE.md`).

## Personas

Mandatory overlay (if present): `~/.grok/personas/_host_overlay_mojo_cuda.md`
Role table: `~/.grok/personas/README_MOJO_CUDA.md`
Prefer **v3** paste (`MOJO_CUDA_PROFILE_V3=1`) when the CLI export is wired.
