# Correctness, Safety, Assembly, and Benchmarking

Last verified: 2026-08-23.

A SIMD implementation is complete only when its semantics, memory safety, feature safety, code generation, and measured performance are all demonstrated.

## 1. Semantic Worksheet

Fill this out before implementation:

```text
Operation:
Input types and valid lengths:
Output type and aliasing rules:
Integer behavior: checked / wrapping / saturating / widening
Float behavior: NaN / signed zero / subnormal / infinity / rounding
Reassociation permitted: yes/no
FMA permitted: yes/no
Approximation and maximum error:
First-error/order requirements:
Minimum target CPU features:
Memory alignment guaranteed:
Overread permitted by API: normally no
Allocation permitted:
```

If any line is unknown, preserve the scalar Rust behavior rather than guessing.

## 2. Scalar Oracle and Differential Tests

Keep the scalar implementation independent from the vector implementation. Do not share the same clever bit trick in both; that can duplicate the same bug.

Test every implementation against the oracle for:

- Lengths `0..(4 * max_vector_lanes + 3)`.
- Exact multiples of every lane count.
- One below and one above every vector boundary.
- Large realistic lengths.
- Misaligned starts produced by subslices at offsets `1..64`.
- Inputs with all equal, alternating, monotonic, random, sparse-match, and dense-match patterns.

For byte algorithms, exhaustive tests over byte values are often affordable. For two-input byte transforms, exhaustively test focused equivalence classes and use property-based random pairs.

## 3. Integer Cases

Include:

- `0`, `1`, and maximum values.
- Values around every widening/narrowing boundary.
- Addition/multiplication overflow.
- Signed minimum (`MIN`) for `abs`-like operations.
- Shift counts `0`, `BITS - 1`, `BITS`, and larger if the API admits them.
- Saturation boundaries exactly one below/at/above the limit.

Never rely on debug-versus-release overflow differences. Encode the intended operation with `wrapping_*`, `saturating_*`, `checked_*`, or a wider accumulator.

## 4. Floating-Point Cases

Include bit-pattern-sensitive cases:

- `+0.0` and `-0.0`.
- Smallest and largest normal values.
- Positive and negative subnormals.
- `+∞` and `-∞`.
- Quiet NaNs with several payloads and signs.
- Values around rounding boundaries.
- Large cancellation and mixed magnitudes for reductions.

Compare according to the contract:

- Bit equality for exact element-wise operations where required.
- Numeric equality when NaN payloads are explicitly irrelevant.
- ULP or relative/absolute error for approximations.
- Class/sign checks for special values.

Do not use a single epsilon for all magnitudes without a numerical rationale.

## 5. Feature-Safety Proof

Every `#[target_feature]` call site needs a local proof.

Acceptable forms:

```rust
if std::arch::is_x86_feature_detected!("avx2") {
    // SAFETY: runtime detection proves AVX2 is supported for this process.
    unsafe { kernel_avx2(...) }
}
```

or a compile-time guarantee:

```rust
#[cfg(target_feature = "avx2")]
{
    // The whole containing code path is built with AVX2 enabled.
}
```

Runtime detection is usually required for distributable binaries. Do not let a feature-specific function escape through an unchecked callback or function pointer.

### Implication boundaries

Use only documented implications. For example, AVX2 implies AVX in Rust's target-feature model, but AVX alone does not imply AVX2. Optional Arm extensions such as dot product are separate from baseline NEON.

## 6. Memory-Safety Proof Template

For every unsafe load/store, establish:

```text
Pointer origin:
Number of initialized bytes/elements available:
Number of bytes/elements loaded or stored:
Alignment required by the intrinsic:
Why alignment is satisfied, or why an unaligned intrinsic is used:
Why pointer arithmetic stays within the same allocated object:
Why output writes do not overlap incompatible live references:
Why all loaded bit patterns are valid for the vector element type:
Tail handling:
```

Prefer deriving a vectorized prefix using safe slice operations before taking pointers. Example:

```rust
let vector_len = input.len() / LANES * LANES;
let (head, tail) = input.split_at(vector_len);
```

Then prove each loop offset is less than `head.len()` and advances by exactly `LANES`.

### No speculative overread

A load beyond a slice is invalid even if:

- The allocator reserved extra capacity.
- The next page is mapped.
- The bytes are never used.
- The CPU would normally tolerate it.

Padding is safe only when initialized storage and its accessible extent are part of the API contract.

## 7. Aliasing and In-Place Operations

Safe Rust slices help LLVM reason about disjoint borrows, but manually created pointers can reintroduce aliasing uncertainty or undefined behavior.

State whether:

- Output can equal one input exactly.
- Inputs can overlap partially.
- Forward or backward processing is required.
- The operation supports repeated writes from an overlapping-tail strategy.

Do not construct simultaneous `&mut` and `&` references to overlapping memory. When in-place operation is required, design the API and loop order deliberately.

## 8. Assembly Verification

Build the exact target/profile being evaluated:

```bash
cargo rustc --release --lib -- --emit=asm
cargo rustc --release --lib -- --emit=llvm-ir
```

For a specific target:

```bash
cargo rustc --release --target x86_64-unknown-linux-gnu --lib -- --emit=asm
cargo rustc --release --target aarch64-unknown-linux-gnu --lib -- --emit=asm
```

Feature-specific experiment builds:

```bash
RUSTFLAGS='-C target-feature=+sse2' cargo rustc --release -- --emit=asm
RUSTFLAGS='-C target-feature=+avx2' cargo rustc --release -- --emit=asm
RUSTFLAGS='-C target-cpu=native' cargo rustc --release -- --emit=asm
```

Use `target-cpu=native` only as a local experiment unless deployment CPUs are homogeneous and controlled.

Inspect:

- The vectorized main loop.
- Trip-count/versioning guards.
- Tail path.
- Loads/stores and their widths.
- Unexpected calls, spills, scalar lane extraction, or repeated broadcasts.
- Runtime detection placement.
- `vzeroupper`/transition-related code on relevant x86 code paths.
- Whether 256-bit code is genuinely doing twice the useful work rather than adding shuffles.

A grep for `xmm`, `ymm`, or `vadd` is not sufficient. Read the loop.

## 9. Auto-Vectorization Evidence

Compare at least:

1. Portable scalar release build.
2. Target-tuned scalar build.
3. Explicit SIMD implementation.

LLVM's Loop and SLP vectorizers are enabled in normal optimized configurations, but the cost model may reject a loop. Do not call a loop “auto-vectorized” without assembly/IR evidence.

Beware `opt-level = "z"`: size optimization can disable loop vectorization. Record the complete Cargo profile.

## 10. Benchmark Design

Measure the behavior that matters to the application.

### Minimum benchmark matrix

- Tiny: `0..64` elements.
- Small: cache-resident inputs.
- Medium: representative working set.
- Large: memory-bandwidth regime.
- Real corpus/distribution.
- Adversarial branch/match distribution.

### Environment record

```text
CPU model and microarchitecture:
OS/kernel:
Rust toolchain and commit/date:
Target triple:
RUSTFLAGS:
Cargo profile, LTO, codegen-units:
Crate versions/features:
Input sizes/distribution:
Warm-up and sample count:
Thread pinning/frequency policy if controlled:
```

### Prevent accidental elimination

Use results in an observable way. `std::hint::black_box` is useful for benchmark inputs/outputs, but it is only a best-effort optimization barrier and is not a cryptographic or correctness boundary.

### Separate latency and throughput

- Latency benchmark: one dependent operation or one request at a time.
- Throughput benchmark: enough independent work to fill execution units.
- End-to-end benchmark: include dispatch, allocation, parsing, or I/O when the real workload does.

A kernel-only win can disappear when dispatch or format conversion dominates.

## 11. Performance Counters

Where available, use platform tools to explain results, not just report time:

- Linux `perf stat` / `perf record`.
- macOS Instruments / `xctrace`.
- Windows Performance Analyzer or a suitable profiler.

Useful counters/signals include cycles, instructions, IPC, branch misses, cache misses, memory bandwidth, and frontend stalls. Counter names and reliability are platform-specific.

## 12. Common False Wins

- Benchmark input or result optimized away.
- Release SIMD compared with debug scalar code.
- `target-cpu=native` SIMD compared with generic scalar code.
- Different overflow/NaN semantics.
- SIMD implementation skips validation or tail work.
- Input is always aligned in the benchmark but not in production.
- One CPU model used to justify every x86/Arm deployment.
- Hot data remains in L1 in a synthetic benchmark but is cold in the application.
- Benchmark excludes dispatch or layout conversion that production pays.
- Mean reported without variance/outliers.
- Wider AVX code wins throughput but increases latency, code size, or power enough to hurt the application.

## 13. Code Size and Multiversioning

Inspect executable/library size when adding several variants. `fearless_simd` and manual dispatch can instantiate each inlined call graph multiple times. Mitigations that may help include:

- Keep dispatched kernels small.
- Move feature-independent work outside them.
- Use LTO.
- Experiment with `codegen-units = 1`.
- Remove variants that are not measured winners.

Do not assume LTO always improves runtime; benchmark it and record build-time tradeoffs.

## 14. Review Output Template

An LLM agent should finish SIMD work with:

```text
Hotspot and workload:
Scalar oracle semantics:
Selected technique and why:
Target features and dispatch proof:
Memory/tail safety proof:
Assembly observations:
Correctness tests added:
Benchmark setup:
Results by input size and CPU:
Regressions/tradeoffs:
Fallback and portability status:
Remaining uncertainty:
```

Use “expected”, “likely”, or “requires measurement” when evidence is absent. Never fabricate benchmark numbers or claim a specific instruction without inspection.
