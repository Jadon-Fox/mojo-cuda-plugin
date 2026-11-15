# mojo-cuda

Grok Build **specialty unit**: Mojo + CUDA + MLIR coding profile **v3**
(Mojo **1.0** / MAX **26.5**, `max.gpu.*`, Ampere **sm_86**).

## What this is

| Piece | Role |
| --- | --- |
| **Skill** `mojo-cuda` | How the agent writes/reviews GPU kernels (altitude, AI-visible patterns, disk evidence, Mojo 1.0 law) |
| **Agents** | implementer · reviewer · entry-stack |
| **Refs** | v3 host law, import migration, kernel-patterns, mlir-evidence, GPU-pilot leaf, optional host-runtime |

## What this is not

- Not a training orchestrator
- Not product train/serve green by itself
- Not a workbench vector-add as orch progress
- Not `training_verified` / `measured_viability` / Unsloth-beat

## Install (local path)

```bash
grok plugin install /home/jfox/src/mojo-cuda-plugin --trust
```

Reload plugins in-session (`r` on Plugins tab) or start a new session.

## Skill already on host

Portable skill text is synced with `~/.grok/skills/mojo-cuda/`. After plugin install,
prefer one source of truth (plugin or user skill) so texts do not drift — re-copy from
this repo’s `skills/mojo-cuda/` when updating.

## GPU pilot (host)

Product leaf is **batched RMSNorm** under `max.gpu.host.DeviceContext` + `std.gpu`
thread ids — not vector-add. CUDA twin: `launch_rms_norm_batch` in orch
`mojo-kernels/cuda/rms_norm_batch_ffi.cu`.

On the GPU host (40-Runtime-Deps / Ampere 3060):

```bash
cd /home/workspace/training_orchestrator
bash mojo-kernels/scripts/smoke_max_gpu_rms.sh
```

`MOJO_GPU_PILOT_RMS_OK` is a **compile+golden** token, not training_verified.

## Codename / version

- Codename (workbench lineage): `grok-build-mojo-cuda`
- Plugin version: **3.1.0**
- Profile: **v3** (supersedes v2 paste; v2 law links still valid)

## License

MIT — see [LICENSE](./LICENSE).
