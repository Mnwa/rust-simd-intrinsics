---
name: rust-simd-intrinsics
license: MIT
description: >-
  Design, implement, review, debug, benchmark, and port performance-critical Rust SIMD code across x86/x86_64 SSE (SSE1), SSE2, AVX and AVX2, and Arm NEON. Use for auto-vectorization, runtime feature dispatch, std::arch intrinsics, fearless_simd, wide, or nightly std::simd; and for byte scanning, compression matchers, image/audio/DSP kernels, reductions, masks, widening, saturation, and vectorized transforms. Do not trigger for ordinary non-hot code where SIMD has not been justified by profiling.
metadata:
  version: "1.1.0"
  last-verified: "2026-09-25"
---

# Rust SIMD Intrinsics

## Compatibility

Stable Rust supports `std::arch`, `wide`, and `fearless_simd` (v1 requires Rust 1.89+). Nightly Rust with `#![feature(portable_simd)]` is required for `std::simd`. Performance conclusions require benchmarks on the actual target CPUs.

## Mission

Produce Rust SIMD code that is faster on measured workloads, correct for every input admitted by the API, safe at architecture boundaries, and maintainable across target triples. Prefer the highest-level technique that produces the required code generation. Never trade away semantic correctness or target safety for an unverified speed claim.

## Read References Progressively

Load only the references needed for the task:

- Read `references/platforms-and-instructions.md` before selecting SSE, SSE2, AVX, AVX2, or NEON intrinsics.
- Read `references/auto-vectorization.md` before replacing scalar Rust with explicit vectors.
- Read `references/libraries.md` when choosing among `fearless_simd`, `wide`, nightly `std::simd`, and `std::arch`.
- Read `references/algorithm-cookbook.md` for predefined kernels and instruction mappings.
- Read `references/correctness-benchmarking.md` before claiming a speedup or merging unsafe code.
- Read `references/source-index.md` when an API, crate version, target baseline, or instruction fact may have changed.
- Use `examples/` as implementation templates, not as universal benchmark winners.

## Non-Negotiable Rules

1. Start with a scalar reference implementation and tests.
2. Profile first. Identify the hot loop, representative data sizes, target CPUs, and whether the kernel is compute-, branch-, latency-, or memory-bound.
3. Define semantics before vectorizing: integer overflow, NaNs, signed zero, subnormals, rounding, saturation, reduction order, aliasing, and permitted overreads.
4. Dispatch once outside the hot loop. Never run feature detection for each element or chunk.
5. Every `#[target_feature]` function must be called only after compile-time or runtime proof that the feature exists. Calling it on unsupported hardware is undefined behavior.
6. Keep a scalar fallback unless the public deployment contract guarantees a stronger baseline.
7. Use unaligned loads by default. Use aligned loads only when alignment is mechanically guaranteed and documented.
8. Never read or write outside a slice, even when the hardware load would probably stay inside a mapped page.
9. Handle tails explicitly. Scalar cleanup is the default.
10. Do not assume one intrinsic becomes one instruction. Inspect optimized assembly.
11. Do not assume wider vectors are faster. Benchmark AVX against SSE2/NEON-width alternatives on the deployment CPUs.
12. Do not use `#[inline(always)]` together with `#[target_feature]`. Rust forbids this combination. `fearless_simd` generic kernels are a separate case and follow that crate's inlining rules.
13. Do not enable `target-cpu=native`, `+avx`, or `+avx2` for a distributable binary unless every deployment machine is guaranteed to support it.
14. Do not use `unsafe`, unchecked indexing, manual pointer arithmetic, prefetching, non-temporal stores, or hand unrolling merely because they look low-level. Require evidence.
15. Performance claims must include the benchmark setup and must not be invented.

## Technique Selection Ladder

Choose the first level that satisfies the measured requirement:

1. **Scalar Rust shaped for LLVM auto-vectorization.** Best maintenance and often enough for contiguous element-wise loops.
2. **`fearless_simd`.** Prefer when safe runtime multiversioning, native-width portable vectors, or safe access to raw intrinsics is valuable.
3. **`wide`.** Prefer for stable, fixed-width explicit vectors and straightforward portable arithmetic when compile-time target selection is acceptable.
4. **Nightly `std::simd`.** Prefer when standard-library portable semantics, masks, gathers/scatters, or masked tails justify nightly Rust.
5. **`std::arch`.** Use for a proven architecture-specific hot path, specialized instructions, exact instruction selection, or algorithms not expressed efficiently by a portable API.

Do not jump directly to intrinsics. Compare at least the scalar optimized build and the selected explicit SIMD implementation.

## Required Workflow

### 1. Establish the Contract

State or infer explicitly:

- Input/output types and lengths.
- Whether slices may alias.
- Typical and worst-case sizes.
- Required target triples and minimum CPU features.
- Exact versus approximate floating-point behavior.
- Overflow and saturation behavior.
- Alignment guarantees.
- Whether allocation is allowed.
- Whether `std`, nightly Rust, or third-party crates are allowed.

When requirements are absent, default to stable Rust, no allocation in the kernel, no overreads, unaligned memory access, scalar tails, runtime dispatch, and exact element-wise semantics.

### 2. Build a Scalar Oracle

Write the simplest correct implementation. Make overflow operations explicit with `wrapping_*`, `saturating_*`, or checked arithmetic. For floating point, state whether reassociation, FMA contraction, approximate reciprocal, flush-to-zero, and non-deterministic reduction order are permitted.

### 3. Create Adversarial Tests

Test lengths around every vector boundary: `0`, `1`, `LANES-1`, `LANES`, `LANES+1`, `2*LANES-1`, `2*LANES`, and larger odd lengths. Add domain-specific edge cases:

- Integer minimum/maximum and overflow.
- NaNs, infinities, `+0.0`, `-0.0`, and subnormals when relevant.
- Misaligned subslices such as `&data[1..]`.
- Empty and very small inputs.
- First/last-lane matches and mismatches.
- Aliasing scenarios permitted by the API.

### 4. Try Auto-Vectorization First

Shape the loop as contiguous, independent, side-effect-free work. Assert equal lengths once, keep branches simple, avoid opaque calls inside the loop, and inspect optimized assembly. Read `references/auto-vectorization.md`.

### 5. Select ISA and Abstraction

Use the target matrix in `references/platforms-and-instructions.md`.

- SSE (often called SSE1 to distinguish it from SSE2) is primarily the 128-bit `f32` baseline being requested here.
- SSE2 adds 128-bit integer and `f64` operations.
- AVX adds 256-bit floating-point vectors and VEX encodings.
- AVX is not AVX2. General 256-bit integer arithmetic such as `_mm256_add_epi32` requires AVX2.
- NEON uses 64- and 128-bit vectors; AArch64 normally exposes 128-bit AdvSIMD/NEON as a platform baseline on mainstream hard-float OS targets, while Armv7 NEON is optional.

### 6. Isolate Feature-Specific Kernels

Use a safe public wrapper and private feature-specific functions:

```rust
pub fn kernel(args: ...) -> Output {
    if feature_is_available() {
        // SAFETY: runtime detection proves the function's target features.
        return unsafe { kernel_feature(args) };
    }
    kernel_scalar(args)
}

#[target_feature(enable = "...")]
unsafe fn kernel_feature(args: ...) -> Output {
    // Vector loop plus an in-bounds tail.
}
```

Keep the unsafe surface small. Add a safety comment for every unsafe block describing feature, pointer validity, bounds, alignment, initialization, and aliasing proof.

### 7. Implement the Vector Loop

For a lane count `L`:

```text
i = 0
while i + L <= len:
    load exactly L valid elements
    compute lane-wise result
    store exactly L valid elements
    i += L
process [i..len] with scalar code
```

Accumulate reductions in vectors and reduce once after the main loop. Consider two or more independent accumulators only when latency or dependency chains are measured bottlenecks.

### 8. Verify Code Generation

Build with release optimization and inspect assembly or LLVM IR. Confirm:

- The intended vector instructions exist.
- No scalar work remains in the main loop unexpectedly.
- No repeated feature detection occurs.
- Bounds checks, spills, conversions, and shuffles are reasonable.
- The tail is in-bounds.
- Calls were inlined where intended.
- The selected feature function is not reachable without a feature proof.

Use the scripts in `scripts/` or the commands in `references/correctness-benchmarking.md`.

### 9. Benchmark Correctly

Measure scalar, auto-vectorized, portable SIMD, and architecture-specific variants where relevant. Include tiny, medium, and large inputs; aligned and intentionally offset inputs; realistic and adversarial data; warm and cold-cache scenarios when meaningful. Report median plus dispersion, CPU model, OS, Rust version, target flags, and workload size.

### 10. Return a Complete Result

The final answer or patch must include:

- The scalar reference.
- The optimized implementation.
- Target and dispatch policy.
- Safety invariants.
- Tail strategy.
- Semantic differences, if any.
- Tests.
- Benchmark method and actual results, or a clear statement that performance is not yet measured.
- Assembly evidence or exact commands for obtaining it.

## Dispatch Policy

### Portable Binary

Use runtime detection and per-function `#[target_feature]` kernels. Recommended x86 order for covered features is usually AVX2 for integer kernels, AVX for floating kernels, then SSE2/SSE, then scalar. On AArch64, use NEON when the target baseline or runtime detection proves it.

### Homogeneous Deployment Fleet

A whole-binary baseline such as `-C target-cpu=x86-64-v3` or a named CPU may be reasonable only when deployment rejects incompatible machines. Document the minimum CPU contract and test startup on the oldest supported host.

### Repeated Calls

Cache the selected function pointer or a library dispatch token at a stable outer boundary when dispatch overhead is measurable. Do not introduce global mutable state without need; library-provided cached detection is preferable.

## Correctness Decisions That Must Be Explicit

### Integers

Choose wrapping, saturating, checked, or widening arithmetic. Portable vector libraries may use wrapping operators even when debug scalar Rust would panic. Make both paths intentionally identical.

### Floating Point

Specify:

- Whether operation order may change.
- Whether FMA is required, forbidden, or optional.
- NaN propagation and payload requirements.
- `min`/`max` behavior with NaNs and signed zero.
- Whether approximate reciprocal/square root is allowed.
- Whether subnormal flush-to-zero is acceptable.

Do not vectorize a strict floating-point reduction by silently reassociating it.

### Memory

Unaligned does not mean unchecked. Every load still requires all loaded bytes to belong to valid initialized objects. Aligned intrinsics additionally require their documented alignment.

### Masks

Treat mask representation as opaque unless the API explicitly defines it. Convert with the library's mask/bitmask operation rather than transmuting. For `wide`, use comparison-produced all-zero/all-one masks with `select`, not arbitrary vector data.

## Library-Specific Rules

### `fearless_simd`

- Target `fearless_simd` 1.0 (Rust 1.89+). Read [the v1 migration and API guidance](references/libraries.md#fearless-v1-migration) when upgrading older code.
- Prefer `#[simd]` from the separate optional `fearless_simd_macros` 0.1 crate for SIMD-generic functions. Without macros, inline small generic kernels into the dispatch context with `#[inline(always)]`, or enter a `vectorize()` context explicitly.
- Enter from scalar code through `dispatch!` or a previously selected `Level`.
- Use `vectorize()` for SIMD-to-SIMD calls when inlining should not be forced.
- Prefer native-width associated vector types such as `S::u32s` when the algorithm scales with vector width; use `LEN` for their lane count.
- Use built-in numeric reducers before composing helpers. FP sum/product have a fixed order for a given vector type and lane count, not across native widths or accumulator layouts.
- Use `mul_add_precise`/`mul_sub_precise` only when correctly rounded fused semantics are required; ordinary multiply then add has a different rounding contract.
- Use `kernel!` only for specialized intrinsics not efficiently represented by portable operations.
- Account for x86 multiversioning code-size growth.

### `wide`

- Treat vector width as an API choice, not a promise that a matching hardware width exists.
- Use explicit full chunks and a scalar remainder.
- Preserve `wide`'s wrapping integer semantics and documented floating-point behavior.
- Use `select` only with valid mask values produced by comparisons or mask constructors.
- Add runtime multiversioning outside `wide` when one binary must exploit optional CPU features.

### Nightly `std::simd`

- Add `#![feature(portable_simd)]` and pin a nightly toolchain in production.
- Expect portable semantics, not a one-intrinsic-to-one-instruction mapping.
- Inspect code generation because an operation may scalarize on a target without efficient support.
- Use masked load/store APIs for tails only after benchmarking them against scalar cleanup.

### `std::arch`

- Gate every architecture module with `cfg`.
- Use feature-specific functions plus runtime detection or a documented compile-time baseline.
- Prefer unaligned load/store intrinsics unless alignment is proven.
- Keep raw pointers inside the smallest possible function.
- Add a scalar oracle and differential tests for every intrinsic implementation.

## Review Checklist

Reject or revise code when any answer is “no”:

- Was the hot path measured?
- Is the scalar behavior documented and tested?
- Is every target feature proved before use?
- Are AVX and AVX2 distinguished correctly?
- Are all loads/stores in bounds and initialized?
- Is alignment correct?
- Are tails correct for all lengths, including zero?
- Are integer and floating-point semantics preserved or intentionally changed?
- Does the code compile for every claimed target?
- Was optimized assembly inspected?
- Was performance measured on representative hardware and data?
- Is the speedup large enough to justify complexity and code size?

## Definition of Done

A SIMD task is complete only when correctness tests pass against the scalar oracle, target-feature safety is proven, the code compiles for the supported target matrix, assembly confirms the intended strategy, and benchmark evidence supports keeping the optimization. Otherwise label the implementation as an unmeasured candidate rather than “optimized.”
