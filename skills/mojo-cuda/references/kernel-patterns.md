# Kernel patterns & anti-patterns

## Prefer

- 1D elementwise with coalesced `base + tid` addressing as the smoke test
- Tiled matmul with shared tiles, barriers, and tail masks
- Online softmax for attention rows (stable max + sum)
- Structured pipelines for production GEMM / fused epilogues
- Host launch sizing: `blocks = (n + threads - 1) // threads`

## Anti-patterns

| Avoid | Prefer |
| --- | --- |
| Unmasked stores past N | `if i < n` / epilogue masks |
| Shared write then read without barrier | `barrier()` between phases |
| Column-strided global loads in warps | Transpose staging or layout change |
| One giant kernel file | Config + pipelines + thin launch |
| Claiming “optimal” without roofline | Name bottleneck + expected gain class |
| Invented Modular imports | Sketches marked with assumptions |

## Review order

1. Correctness / shapes  
2. Global memory behavior  
3. Shared memory reuse & banks  
4. Register pressure  
5. Algorithmic fusion / TC shapes  
