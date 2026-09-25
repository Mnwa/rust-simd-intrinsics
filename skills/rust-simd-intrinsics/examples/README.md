# Rust SIMD Skill Examples

This crate provides small, auditable templates rather than a benchmark library.
Every optimized implementation has a scalar contract, explicit tails, and tests
around vector-width boundaries.

## Modules

- `autovec`: ordinary scalar Rust shaped for LLVM Loop/SLP vectorization.
- `std_arch`: runtime-dispatched SSE (SSE1), SSE2, AVX, AVX2, and AArch64 NEON kernels.
- `wide_impl`: stable fixed-width `wide::f32x8` chunks (`wide-example`).
- `fearless_impl`: safe runtime multiversioning with `fearless_simd` 1.0.0;
  manual inlining and `#[simd]` from `fearless_simd_macros` 0.1.0
  (`fearless-example`).
- `portable_simd`: nightly `std::simd` (`portable-simd-example`).

## Toolchain requirements

- Base `std::arch` and auto-vectorization examples: current stable Rust.
- `fearless-example`: Rust 1.89 or newer, matching the observed `fearless_simd` 1.0.0 MSRV.
- `portable-simd-example`: a nightly toolchain with `portable_simd`.

## Stable checks

```bash
cargo test --release
cargo test --release --features wide-example
cargo test --release --features fearless-example
```

## Nightly portable SIMD

```bash
cargo +nightly test --release --features portable-simd-example
```

## Inspect assembly

From the skill root:

```bash
./scripts/emit-asm.sh generic
./scripts/emit-asm.sh native
./scripts/emit-asm.sh avx2
```

For cross-target assembly, install the target and corresponding linker/toolchain
as required by the host:

```bash
TARGET=aarch64-unknown-linux-gnu ./scripts/emit-asm.sh neon
FEATURES=wide-example ./scripts/emit-asm.sh native
```

## Notes

- `std_arch::add_f32` distinguishes AVX floating-point support from AVX2
  integer support.
- x86 loads are unaligned and still remain strictly in bounds.
- NEON byte search uses vector comparison and horizontal detection, then stores
  only the first matching mask block to recover the lane.
- The examples make no universal speed claim. Benchmark them on the deployment
  CPUs and workload before retaining the added complexity.
