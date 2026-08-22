---
description: Load Mojo+CUDA specialty profile v3 (Mojo 1.0 / max.gpu / Ampere sm_86) for this turn
---

Load skill **mojo-cuda** (v3) and follow its method: altitude → AI-visible review →
hypothesis evidence → merge only if supported. New kernels use
`max.gpu.host.DeviceContext` + `std.gpu` thread ids. GPU pilot leaf is batched
RMSNorm, not vector-add. Use agents mojo-cuda-implementer /
mojo-cuda-reviewer / mojo-cuda-entry-stack when roles should split.
Seals: train_ok=false · measured_omega=false · G1=OPEN.
