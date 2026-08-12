# mojo-cuda

Grok Build **specialty unit**: Mojo + CUDA + MLIR coding profile **v2**.

## What this is

| Piece | Role |
| --- | --- |
| **Skill** `mojo-cuda` | How the agent writes/reviews GPU kernels (altitude, AI-visible patterns, disk evidence) |
| **Agents** | implementer · reviewer · entry-stack |
| **Refs** | system-prompt, kernel-patterns, mlir-evidence, optional host-runtime |

## What this is not

- Not a training orchestrator
- Not the Grok.com Vite workbench UI (optional methodology export elsewhere)
- Not product train/serve green by itself

## Install (local path)

```bash
grok plugin install /home/jfox/src/mojo-cuda-plugin --trust
```

Reload plugins in-session (`r` on Plugins tab) or start a new session.

## Skill already on host

Portable skill text is synced with `~/.grok/skills/mojo-cuda/`. After plugin install,
prefer one source of truth (plugin or user skill) so texts do not drift — re-copy from
this repo’s `skills/mojo-cuda/` when updating.

## Codename / version

- Codename (workbench lineage): `grok-build-mojo-cuda`
- Plugin version: **2.0.0**

## License

MIT — see [LICENSE](./LICENSE).
