# Hub pointer — Mojo+CUDA Specialist v3

**Added:** 2026-08-11
**GPU-pilot leaf:** 2026-08-22
**Profile:** [`MOJO_CUDA_SPECIALIST_V3_HOST.md`](MOJO_CUDA_SPECIALIST_V3_HOST.md)
**Research:** [`MOJO-1-0-MAX-26-5-PROFILE-RESEARCH.md`](MOJO-1-0-MAX-26-5-PROFILE-RESEARCH.md)
**Import migration:** [`AMPERE-SM86-MOJO-1-0-IMPORT-MIGRATION.md`](AMPERE-SM86-MOJO-1-0-IMPORT-MIGRATION.md)
**Pilot:** [`GPU-PILOT-LEAF.md`](GPU-PILOT-LEAF.md)

| Field | Value |
|-------|-------|
| Modular | Mojo 1.0.0 + MAX 26.5 (2026-08-11) |
| Product silicon | RTX 3060 sm_86 12 GiB (dual host: 2080 sm_75) |
| GPU pilot | batched RMSNorm · `rms_norm_max_gpu.mojo` · UNMEASURED until host smoke |
| Plugin | 3.1.0 |
| Seals | train_ok=false · measured_omega=false · G1=OPEN |

CLI: prefer `MOJO_CUDA_PROFILE_V3=1` when export is wired; v2 paste still works with v3 law links.
