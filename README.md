# lite-ssm

Ultra-lightweight standalone C++ inference engine for State Space Models
(starting with Mamba-2). Native to Apple Silicon via Metal + unified memory.
Zero third-party C++ deps.

## Status

Phase 1: project scaffold + CMake/Metal build pipeline. No inference yet.

## Build

```sh
cmake -S . -B build -G Ninja
cmake --build build
ctest --test-dir build
```

Requires macOS with the Xcode command-line tools (provides `xcrun`, `metal`,
`metallib`).

## Layout

See `docs/architecture.md` for the 5-phase plan and SSD strategy.
