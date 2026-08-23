# Auto-Vectorization Patterns for Rust

Last verified: 2026-08-23.

Rust normally lowers optimized code through LLVM. LLVM enables both the Loop Vectorizer and the SLP Vectorizer by default. The Loop Vectorizer widens consecutive loop iterations; the SLP Vectorizer combines independent scalar operations. Their cost model can choose scalar code when vectorization is not profitable.

## 1. Why Start Here

Auto-vectorized scalar code has several advantages:

- One readable implementation remains the semantic source of truth.
- LLVM chooses vector width and unrolling for the target.
- A generic build keeps scalar compatibility.
- A target-feature or `fearless_simd` wrapper can multiversion the same loop.
- The compiler can combine vectorization with surrounding optimizations.

Explicit SIMD is justified when LLVM misses the pattern, selects poor instructions, cannot express the needed semantics, or a specialized instruction gives a measured win.

## 2. Canonical Element-Wise Pattern

```rust
pub fn add_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());

    for ((dst, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *dst = x + y;
    }
}
```

Equivalent index form:

```rust
pub fn add_f32_indexed(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());

    for i in 0..out.len() {
        out[i] = a[i] + b[i];
    }
}
```

Do not assume either syntax is always better. Compile both when code generation matters. Equal-length assertions and ordinary slice references make bounds and aliasing relationships easier for optimization than opaque raw-pointer APIs.

## 3. Patterns That Commonly Vectorize

### Map

```rust
pub fn scale_bias(xs: &mut [f32], scale: f32, bias: f32) {
    for x in xs {
        *x = *x * scale + bias;
    }
}
```

This preserves separate multiply then add semantics at the source level. Do not replace it with `mul_add` unless fused semantics are intended.

### Zip / map2

```rust
pub fn xor_bytes(out: &mut [u8], a: &[u8], b: &[u8]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());

    for ((dst, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *dst = x ^ y;
    }
}
```

### Wrapping integer transform

```rust
pub fn add_u32_wrapping(out: &mut [u32], a: &[u32], b: &[u32]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());

    for ((dst, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *dst = x.wrapping_add(y);
    }
}
```

Use explicit wrapping operations so debug and release builds have the same contract.

### Simple conditional select

```rust
pub fn threshold_in_place(xs: &mut [u8], threshold: u8, high: u8) {
    for x in xs {
        if *x > threshold {
            *x = high;
        }
    }
}
```

LLVM may if-convert simple branches to vector compares and selects. Do not mechanically rewrite every branch as bit tricks; first inspect code generation and branch behavior.

### Integer reduction with explicit wrapping

```rust
pub fn wrapping_sum(xs: &[u32]) -> u32 {
    xs.iter().copied().fold(0, u32::wrapping_add)
}
```

Modulo addition is associative, which gives the optimizer more freedom than strict floating-point addition. Verify assembly; reductions are more sensitive to compiler version and target flags than map kernels.

## 4. Floating-Point Reductions

A scalar floating-point sum has an observable order because IEEE-754 addition is not associative. Vectorizing it usually changes grouping and may change the result.

Do not silently transform:

```rust
pub fn strict_sum(xs: &[f32]) -> f32 {
    xs.iter().copied().fold(0.0, |acc, x| acc + x)
}
```

into a tree or multi-accumulator reduction unless the API permits different rounding.

Define one of these contracts:

- **Strict left-to-right:** retain scalar order; SIMD may not apply to the reduction itself.
- **Deterministic tree:** document a fixed grouping and reproduce it on all paths.
- **Relaxed/fast:** allow reassociation within a documented error tolerance.
- **Compensated:** use Kahan/Neumaier or pairwise summation; vectorize carefully and test error bounds.

Element-wise floating operations are easier because lane independence preserves operation order within each element.

## 5. Patterns That Inhibit Vectorization

Common blockers include:

- Opaque, non-inlined function calls in the loop.
- Trait-object or callback dispatch per element.
- Complicated control flow, early exits, or `match`/`switch` structures.
- Loop-carried dependencies such as `x[i] = x[i - 1] + ...`.
- Potentially overlapping raw pointers.
- Volatile or atomic operations.
- Panics or side effects that must occur at a precise iteration.
- Irregular indexing, pointer chasing, linked structures.
- Expensive conversions or gathers without target support.
- Strict floating-point reduction order.
- Mixed tiny operations where setup and tail dominate.

Do not remove safety checks blindly. First identify the actual blocker in LLVM IR or assembly.

## 6. Express Independence Clearly

Prefer APIs that encode aliasing and ownership:

```rust
pub fn transform(out: &mut [u32], input: &[u32]) {
    assert_eq!(out.len(), input.len());
    for (dst, &src) in out.iter_mut().zip(input) {
        *dst = src.rotate_left(5) ^ 0x9e37_79b9;
    }
}
```

When source and destination are one allocation, split it into disjoint slices with `split_at_mut` instead of constructing multiple raw pointers.

Avoid `get_unchecked` as a first response. LLVM often removes bounds checks from canonical loops. Introduce unchecked indexing only after optimized assembly proves checks remain in the hot loop, a benchmark proves they matter, and the safety invariant is small and testable.

## 7. Function Boundaries and Inlining

Small helper functions can vectorize if they inline. Prefer:

```rust
#[inline]
fn transform_lane(x: u32) -> u32 {
    x.rotate_left(5) ^ 0x9e37_79b9
}
```

Avoid forcing `#[inline(always)]` everywhere. It can increase code size and register pressure. Exceptions:

- `fearless_simd` explicitly requires `#[inline(always)]` for its generic SIMD kernels.
- Tiny helpers proven not to inline and proven to block vectorization may justify it.

`#[target_feature]` functions cannot use `#[inline(always)]`.

## 8. Loop Shape

A good vectorizable loop usually has:

- A single induction variable or straightforward iterator chain.
- Contiguous memory accesses.
- Independent iterations.
- A simple arithmetic body.
- Loop-invariant constants hoisted outside.
- No allocation, logging, locking, or I/O.

Do not hand-unroll before checking LLVM. LLVM's vectorizer has a cost model for vector width and interleave/unroll factors. Manual unrolling can obscure the pattern or increase register pressure.

## 9. Branches and Masks

Simple per-lane conditions can become masks:

```rust
pub fn clamp_u16(xs: &mut [u16], lo: u16, hi: u16) {
    assert!(lo <= hi);
    for x in xs {
        *x = (*x).clamp(lo, hi);
    }
}
```

Check semantic details before replacing standard operations with hardware min/max:

- Integer min/max is straightforward when the ISA has the correct signedness.
- Floating min/max intrinsics may differ in NaN and signed-zero behavior.
- A branch may be faster than mask work when data is highly predictable and the loop is not vectorized.

## 10. Data Layout

Structure of arrays (SoA) is often easier to vectorize than array of structures (AoS):

```text
AoS: [x0,y0,z0, x1,y1,z1, ...]
SoA: x=[x0,x1,...], y=[y0,y1,...], z=[z0,z1,...]
```

Do not redesign public data layout solely for SIMD without measuring end-to-end effects. Conversion and cache behavior may dominate. On NEON, structured loads can efficiently handle common interleaved layouts such as RGB.

## 11. Build Settings

Use a release profile. A useful measured starting point:

```toml
[profile.release]
opt-level = 3
lto = "thin"
codegen-units = 1
```

Tradeoffs:

- `codegen-units = 1` can improve optimization and reduce duplicate multiversioned code but increases compile time.
- Thin/fat LTO can improve cross-crate inlining at greater build cost.
- `opt-level = "z"` disables loop vectorization; do not use it when SIMD throughput is the goal.
- `opt-level = 2` can occasionally beat `3`; benchmark.
- Incremental compilation can inhibit some release optimizations; use clean release builds for final measurements.

For a local-only benchmark:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Do not confuse this with a portable deployment configuration.

## 12. Inspecting Code Generation

Emit assembly:

```bash
cargo rustc --release --lib -- --emit=asm
```

Emit LLVM IR:

```bash
cargo rustc --release --lib -- --emit=llvm-ir
```

Build for an explicit CPU baseline:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo rustc --release --lib -- --emit=asm
```

Useful checks:

```bash
grep -E "xmm|ymm|vadd|addps|padd|ld1|st1" target/release/deps/*.s
```

Grep is only a first pass. Read the complete loop, labels, trip-count checks, tail, and call boundaries.

For LLVM vectorization diagnostics, Rust exposes some backend arguments only on nightly and they can change. Prefer assembly/IR as the stable verification method. When using nightly diagnostics, record the exact toolchain and command.

## 13. Differential Experiment Strategy

When LLVM does not vectorize:

1. Copy the hot function into a minimal benchmark crate.
2. Keep the scalar oracle unchanged.
3. Simplify one factor at a time:
   - inline a helper;
   - remove a callback;
   - split a branchy loop into two passes;
   - hoist a condition;
   - express disjoint slices;
   - separate rare slow-path values;
   - change AoS to an SoA experiment.
4. Inspect assembly after each change.
5. Benchmark end-to-end, not only the isolated synthetic loop.

## 14. Auto-Vectorization Through `fearless_simd`

`fearless_simd` can multiversion ordinary scalar-looking code:

```rust
use fearless_simd::{dispatch, Level, Simd};

#[inline(always)]
fn double<S: Simd>(_: S, values: &mut [u32]) {
    for value in values {
        *value = value.wrapping_mul(2);
    }
}

pub fn double_best(values: &mut [u32]) {
    let level = Level::new();
    dispatch!(level, simd => double(simd, values));
}
```

The generic SIMD token causes multiple target-feature versions to be generated and selected at runtime. Inspect binary size as well as speed.

## 15. Auto-Vectorization Checklist

Before explicit SIMD, verify:

- Release optimization is enabled.
- `opt-level = "z"` is not disabling loop vectorization.
- The loop is actually hot.
- Inputs are contiguous and iterations independent.
- Length checks are outside the loop.
- Opaque calls and trait dispatch are removed or inlined.
- Arithmetic semantics permit vectorization.
- Assembly was inspected for the actual deployment target.
- Scalar and target-specific builds were benchmarked.
