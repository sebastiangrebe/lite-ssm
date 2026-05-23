# lite-ssm Architecture

## Overview

Ultra-lightweight standalone C++ inference engine for State Space Models,
starting with Mamba-2. Apple Silicon only. Zero third-party C++ deps.

## Phases

1. **Scaffold & build system** — CMake, Metal toolchain, placeholder tree.
2. **Python weight exporter** — HuggingFace Mamba-2 → custom `.ssm` binary.
3. **Core C++ tensor + memory** — `mmap` + Metal `newBufferWithBytesNoCopy`
   for zero-copy unified-memory weight access.
4. **Metal kernels** — Linear, RMSNorm, SiLU, causal conv1d, chunked SSD
   (prefill), fused SSD step (decode).
5. **Fixed-state inference loop** — no growing KV cache; fixed-size
   recurrent state updated in place per token.

## SSD strategy (Phase 4 preview)

- **Prefill:** chunked SSD per Tri Dao paper. Per-chunk intra-block matmul
  in threadgroup SRAM; inter-chunk state pass via Blelloch prefix scan over
  `N/C` chunk-final states. `simdgroup_matrix_multiply` on M3+.
- **Decode:** `L=1`; collapses to single fused kernel
  `s ← decay·s + B·x; y ← C·s + D·x`.
