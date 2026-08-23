# Rust SIMD Library Selection and Patterns

Last verified: 2026-08-23.

This reference compares the four implementation layers used by this skill. Version numbers are observation points, not permanent constraints. Re-check `references/source-index.md` before pinning dependencies.

## 1. Decision Table

| Need | First choice | Why | Main caveat |
|---|---|---|---|
| Readable hot loop that LLVM can widen | Scalar Rust | Smallest maintenance cost | Code generation must be inspected |
| Stable Rust plus safe runtime multiversioning | `fearless_simd` | Safe dispatch token, native-width abstractions, scalar fallback | Multiversioning increases compile time and code size |
| Stable fixed-width explicit vector arithmetic | `wide` | Simple concrete vector types | It does not by itself create a runtime AVX/SSE/NEON dispatch boundary |
| Portable masks, gather/scatter, masked memory | Nightly `std::simd` | Standard portable semantics and broad API | Experimental nightly API; code can scalarize |
| Specialized instructions or exact architecture control | `std::arch` | Direct architecture intrinsics | Unsafe boundary, duplicated kernels, manual dispatch |

Use only one abstraction inside a kernel unless an instruction unavailable in the abstraction justifies a small, isolated escape hatch.

## 2. `fearless_simd`

Observed crate version: `0.7.0`.

### Best fit

Use `fearless_simd` when the binary must run on a range of CPUs and the kernel benefits from selecting the best supported level at runtime. It supports safe SIMD abstractions, auto-vectorized generic kernels, explicit vector types, and access to raw intrinsics through architecture tokens.

Current documented implementation levels include:

- Scalar fallback.
- x86/x86_64 SSE2 baseline.
- x86-64-v2 / SSE4.2-class level.
- x86-64-v3 / AVX2-class level.
- AArch64 NEON.
- Additional higher x86 levels where the crate and target support them.

Do not infer that every operation exists at every level. The trait bounds and generated vector types remain the API contract.

### Pattern A: multiversioned scalar-looking loop

```rust
use fearless_simd::{dispatch, Level, Simd};

#[inline(always)]
fn double_kernel<S: Simd>(_: S, values: &mut [u32]) {
    for value in values {
        *value = value.wrapping_mul(2);
    }
}

pub fn double(values: &mut [u32]) {
    let level = Level::new();
    dispatch!(level, simd => double_kernel(simd, values));
}
```

Rules:

- Put `#[inline(always)]` on the generic kernel as recommended by the crate so it is specialized inside each dispatched version.
- Keep `Level::new()` and `dispatch!` outside the hot loop.
- Use explicit overflow operations in scalar-looking integer code.
- Inspect binary size. Each dispatched version duplicates code.

### Pattern B: fixed-width explicit vectors with an explicit scalar tail

```rust
use fearless_simd::{dispatch, prelude::*, u32x4, Level};

#[inline(always)]
fn double_explicit<S: Simd>(simd: S, values: &mut [u32]) {
    let (chunks, tail) = values.as_chunks_mut::<4>();

    for chunk in chunks {
        let x = u32x4::load_array_ref(simd, chunk);
        (x + x).store_array(chunk);
    }

    for value in tail {
        *value = value.wrapping_mul(2);
    }
}

pub fn double(values: &mut [u32]) {
    let level = Level::new();
    dispatch!(level, simd => double_explicit(simd, values));
}
```

`as_chunks_mut` exposes full chunks as `&mut [u32; 4]`, which is the exact
array type accepted by `u32x4::load_array_ref` and `store_array`. Each full
chunk is processed explicitly as SIMD; the tail stays ordinary scalar Rust.

The analogous native-width loader is `S::u32s::load_array_ref`. Its array
length varies with `S`, so standard `slice::as_chunks` cannot express that
length in a generic function on stable Rust. Use `chunks_exact` plus
`S::u32s::from_slice` when native width is more important than a fixed-array
boundary.

### Pattern C: dispatch once and reuse

When several kernels run in a batch, detect the level once and pass it down rather than repeatedly constructing a dispatch boundary. Prefer an API such as:

```rust
pub fn process_batch(/* ... */) {
    let level = fearless_simd::Level::new();
    // Dispatch a coarse-grained operation that calls several inlined kernels.
}
```

### Pitfalls

- A generic `S: Simd` function that is not inlined may lose specialization opportunities.
- Multiversioning a very large call graph can cause code-size and instruction-cache regressions.
- `Level::new()` chooses a supported implementation level; it does not prove that a particular raw intrinsic extension beyond that level is available.
- Do not cache a process-wide level in complicated unsafe initialization unless measurement shows construction matters. Prefer simple ownership and dispatch first.

## 3. `wide`

Observed crate version: `1.6.1`.

### Best fit

Use `wide` for stable Rust when a concrete lane count is natural and the operation maps well to its vector types. Examples include `f32x4`, `f32x8`, `i16x8`, `u8x16`, and larger fixed-width types.

The crate uses explicit SIMD where possible. On unsupported targets, LLVM vectorization or scalar code can remain correct. That portability is not the same as guaranteed native SIMD performance.

### Fixed-width map pattern

```rust
use wide::f32x8;

pub fn add_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());

    let (out_chunks, out_tail) = out.as_chunks_mut::<8>();
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, b_tail) = b.as_chunks::<8>();

    for ((dst, x), y) in out_chunks
        .iter_mut()
        .zip(a_chunks)
        .zip(b_chunks)
    {
        let xv = f32x8::new(*x);
        let yv = f32x8::new(*y);
        *dst = (xv + yv).to_array();
    }

    for ((dst, &x), &y) in out_tail.iter_mut().zip(a_tail).zip(b_tail) {
        *dst = x + y;
    }
}
```

The full arrays are the SIMD path and the remainder slices are deliberately
scalar. Check assembly: the array conversions normally optimize away, but this
is an optimization result rather than a source-level guarantee.

### Single-block `first_chunk` pattern

Use `first_chunk` when only the first complete block should be vectorized and
all remaining elements should stay scalar:

```rust
use wide::f32x8;

pub fn add_first_block(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());

    let vectorized = match (
        out.first_chunk_mut::<8>(),
        a.first_chunk::<8>(),
        b.first_chunk::<8>(),
    ) {
        (Some(dst), Some(x), Some(y)) => {
            *dst = (f32x8::new(*x) + f32x8::new(*y)).to_array();
            8
        }
        _ => 0,
    };

    for ((dst, &x), &y) in out[vectorized..]
        .iter_mut()
        .zip(&a[vectorized..])
        .zip(&b[vectorized..])
    {
        *dst = x + y;
    }
}
```

### Mask/select pattern

```rust
use wide::f32x8;

pub fn clamp_block(x: f32x8, lo: f32x8, hi: f32x8) -> f32x8 {
    x.max(lo).min(hi)
}
```

Before using `min`, `max`, `fast_min`, `fast_max`, or fused operations, read the crate's NaN and signed-zero semantics. They may differ from a scalar expression you are replacing.

### Runtime portability

A generic binary using `f32x8` cannot blindly execute AVX instructions on an old x86 CPU. The exact implementation depends on the crate, compiler target, and enabled target features. When runtime CPU heterogeneity matters:

1. Prefer `fearless_simd`; or
2. Place separate `wide` kernels behind correctly guarded `#[target_feature]` functions and inspect the emitted code; or
3. Use a conservative baseline vector type and benchmark it.

Never assume the type name alone establishes a safe runtime dispatch policy.

### Semantic cautions

- Integer arithmetic operators are documented as wrapping, including debug builds. Match the scalar oracle explicitly.
- Floating-point reduction order is not a deterministic left fold.
- `mul_add` can fuse on some targets and not others, changing rounding.
- NaN payload bit patterns are not a stable cross-platform contract.
- Convenience conversions from slices may pad, reject, or panic depending on the API. For hot loops, use exact chunks and explicit tails.

## 4. Nightly `std::simd`

Observed nightly documentation: Rust `1.100.0-nightly` dated 2026-08-22. The API is still gated by `portable_simd`.

### Setup

```rust
#![feature(portable_simd)]

use std::simd::Simd;
```

Keep nightly-only code behind a crate feature or a separate crate when stable consumers exist.

### Fixed-lane map pattern

```rust
#![feature(portable_simd)]

use std::simd::Simd;

type F32x8 = Simd<f32, 8>;

pub fn add_f32(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());

    let (out_chunks, out_tail) = out.as_chunks_mut::<8>();
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, b_tail) = b.as_chunks::<8>();

    for ((dst, x), y) in out_chunks.iter_mut().zip(a_chunks).zip(b_chunks) {
        let x = F32x8::from_array(*x);
        let y = F32x8::from_array(*y);
        *dst = (x + y).to_array();
    }

    for ((dst, &x), &y) in out_tail.iter_mut().zip(a_tail).zip(b_tail) {
        *dst = x + y;
    }
}
```

`as_chunks` makes the vectorized arrays and scalar tail distinct in the type
system, so every full array is processed as SIMD without indexing a slice
window.

### Masked-tail pattern

`std::simd` provides masked load/store operations. Use them only after checking their current signatures and code generation. A scalar tail is often simpler and can be faster for one final partial vector.

### Portable does not mean equally fast

The API defines portable operations and layouts, but unsupported operations can be expanded into instruction sequences or scalarized. Always inspect each target. Be especially cautious with:

- Gather/scatter.
- 64-bit integer multiply on targets without a direct vector instruction.
- Variable shifts.
- Narrow types with unusual lane counts.
- Transcendentals and library calls.
- Mask conversions and horizontal reductions.

### Floating-point portability

Portable SIMD aims for consistent semantics, but documented target exceptions can exist. Current documentation notes older Armv7 and PowerPC handling of subnormal `f32` values. Treat subnormals as part of the contract when reproducibility matters.

## 5. `std::arch`

### Best fit

Use `std::arch` when the algorithm requires a specialized instruction, portable abstractions produce poor code, or exact architecture control gives a measured win. Typical examples:

- SSE2 `_mm_sad_epu8` for byte sum-of-absolute-differences.
- SSE2/AVX2 byte equality plus movemask for scanning.
- NEON structured loads for RGB/RGBA deinterleaving.
- Saturating pack/narrow sequences.
- Architecture-specific table lookup or shuffle strategies.

### Required shape

```rust
pub fn operation(/* ... */) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the runtime check proves AVX2 support.
            return unsafe { operation_avx2(/* ... */) };
        }
    }

    operation_scalar(/* ... */)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn operation_avx2(/* ... */) {
    // Keep unsafe memory operations in small, reviewed blocks.
}
```

For AArch64, baseline NEON is generally available for standard AArch64 targets, but keep architecture modules behind `cfg`. Optional features such as dot product still need their own compile-time or runtime proof.

### Intrinsic API rules

- Import only the architecture module selected by `cfg`.
- Keep one feature set per kernel name: `_sse2`, `_avx2`, `_neon`.
- Do not put `#[inline(always)]` on a `#[target_feature]` function.
- Keep pointer validity and loaded byte ranges easy to audit.
- Use `_loadu_` on x86 unless alignment is proven.
- Add a `// SAFETY:` comment at every call and memory access.
- Inspect the full loop and tail in optimized assembly.

## 6. Combining Libraries Deliberately

Good combinations:

- Scalar oracle + auto-vectorized production loop.
- Scalar oracle + `fearless_simd` dispatch + explicit native-width kernel.
- Scalar oracle + `wide` implementation for a fixed deployment target.
- Portable `std::simd` implementation plus one small `std::arch` specialization for a unique instruction.

Usually poor combinations:

- Converting between `wide`, `std::simd`, and architecture vector types inside every iteration.
- Using `fearless_simd` dispatch around a function already performing its own runtime feature detection.
- Maintaining SSE, SSE2, AVX, AVX2, NEON, `wide`, and `std::simd` versions without benchmark evidence for each.

Every additional implementation multiplies testing, code size, and review cost. Keep only measured winners or strategically important fallbacks.

## 7. Library Review Checklist

- Is the dependency/version current and compatible with the project MSRV?
- Does the chosen library perform runtime dispatch, compile-time selection, or neither?
- Are lane count and tail semantics explicit?
- Are integer overflow and float edge cases matched to the oracle?
- Does the target actually lower operations to SIMD?
- Is feature detection outside the hot loop?
- Has code size been compared after multiversioning?
- Are nightly and optional dependencies isolated behind features?
