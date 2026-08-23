# Platforms and Instruction Sets

Last verified: 2026-08-23.

Use this reference before selecting `std::arch` intrinsics or declaring a minimum CPU baseline. Target triples, compiler versions, and deployment rules can change; re-check the authoritative links in `source-index.md` when shipping.

## 1. Mental Model

SIMD performance comes from processing independent lanes with fewer decoded instructions and less loop overhead. It does not remove memory bandwidth, cache misses, dependencies, branch unpredictability, or conversion costs. The best vector width depends on the algorithm and CPU.

Keep these dimensions separate:

- **Architecture:** x86/x86_64 versus Arm/AArch64.
- **ISA extension:** SSE, SSE2, AVX, AVX2, NEON, and optional later extensions.
- **Register width:** 128 bits for XMM; 256 bits for YMM; 64/128 bits for NEON.
- **Element type:** bytes, words, integers, `f32`, or `f64`.
- **Compile-time baseline:** features enabled for the whole crate/target.
- **Runtime availability:** features detected on the machine running the binary.
- **OS support:** relevant for extended x86 state; prefer Rust's detection macros instead of hand-written CPUID logic.

## 2. Capability Matrix

| ISA | Vector registers used here | Core strengths | Important limitations |
|---|---:|---|---|
| SSE (SSE1) | 128-bit XMM | Packed/scalar `f32`, compares, masks, shuffles, logical operations | Do not treat SSE as a complete packed-integer ISA. Use SSE2 for the integer kernels covered by this skill. |
| SSE2 | 128-bit XMM | Packed integers, `f64`, integer compares, shifts, saturating byte/word arithmetic, byte movemask, SAD | No general variable per-lane shifts; many byte rearrangements are awkward without later extensions. |
| AVX | 256-bit YMM for floating point | 256-bit `f32`/`f64`, VEX-encoded three-operand forms, more register freedom | AVX is not AVX2. General 256-bit integer arithmetic remains unavailable. Many cross-128-bit-lane rearrangements need care. |
| AVX2 | 256-bit YMM integer and floating point | General 256-bit integer arithmetic, byte compares/movemask, gathers, wider integer kernels | Still has 128-bit lane boundaries for several shuffle/unpack operations. Wider code can increase pressure and is not automatically faster. |
| Arm NEON / AdvSIMD | 64-bit and 128-bit vectors | Integer and floating arithmetic, widening/narrowing, saturation, pairwise reductions, table operations, structured interleaved loads/stores | AArch32 and AArch64 intrinsic availability differs. Optional extensions such as dot product must be gated separately. No x86-style universal byte movemask instruction. |

## 3. Platform Baselines

### x86_64

- SSE2 is part of the architectural baseline normally assumed by x86_64 Rust targets.
- SSE is therefore also available when SSE2 is available.
- AVX and AVX2 remain optional for portable binaries and require runtime detection or a stronger deployment baseline.
- A custom target specification can change assumptions. Inspect `rustc --print cfg --target <triple>` and `rustc --print target-features --target <triple>`.

### 32-bit x86

- Do not infer features from the word `i686` alone in generic discussions.
- Current mainstream Rust Tier 1 i686 targets are documented as Pentium 4 class, but custom targets and embedded environments may differ.
- Use `is_x86_feature_detected!` in `std` binaries when the feature is optional.

### AArch64

- Mainstream hard-float AArch64 OS target specifications generally include NEON/AdvSIMD as a baseline.
- Custom soft-float or unusual bare-metal targets can be exceptions.
- Prefer `cfg!(target_feature = "neon")` for a known compile-time baseline and `is_aarch64_feature_detected!("neon")` when the target/runtime combination requires detection.
- Optional extensions such as `dotprod`, `fp16`, `bf16`, AES, and SHA are separate features.

### Armv7 / AArch32

- NEON is optional at the architecture/platform level.
- Gate NEON code and preserve a scalar fallback unless the target triple and deployment contract guarantee NEON.
- Some Arm intrinsics differ in stabilization or module path between `core::arch::arm` and `core::arch::aarch64`.

### Big-endian targets

- Do not assume lane numbering, byte reinterpretation, and memory layout observations from little-endian machines apply unchanged.
- Prefer numeric load/store intrinsics and library abstractions over byte transmutation.
- Add target-specific tests if big-endian support is claimed.

## 4. Register Files and ABI Boundaries

- 32-bit x86 exposes eight XMM/YMM register names; x86_64 exposes sixteen for SSE/AVX code.
- AArch64 exposes thirty-two 128-bit `V` registers used by floating-point and AdvSIMD/NEON operations.
- The operating-system ABI decides which vector registers are caller- or callee-saved. This differs across x86 SysV, Windows x64, and AAPCS64.
- Let Rust/LLVM manage those conventions. Do not hand-carry values across calls or inline assembly without reading the exact ABI.
- Keep public APIs in slices, arrays, and scalar types. Architecture vector types make poor stable FFI boundaries and can couple callers to a feature-specific ABI.
- Calls inside a tight vector loop can force spills or inhibit vectorization even when they are semantically cheap. Inline or hoist them only after inspection.

### AArch32 NEON dispatch

On stable Rust, prefer a target specification that guarantees NEON, a separate NEON build selected by the application/installer, or a maintained dispatch crate. As of Rust 1.98, `std::arch::is_arm_feature_detected!` exists but is still nightly-only under `stdarch_arm_feature_detection`.

A nightly-only skeleton is:

```rust
#![feature(stdarch_arm_feature_detection)]

#[cfg(target_arch = "arm")]
pub fn kernel(values: &mut [u32]) {
    if std::arch::is_arm_feature_detected!("neon") {
        // SAFETY: runtime feature detection proves NEON support.
        unsafe { kernel_neon(values) }
    } else {
        kernel_scalar(values)
    }
}

#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon")]
unsafe fn kernel_neon(values: &mut [u32]) {
    use core::arch::arm::*;
    // Full in-bounds vector loop plus scalar tail.
}
```

For a target whose specification guarantees NEON, compile-time `cfg(target_feature = "neon")` may replace runtime detection. Do not assume this from the `arm` architecture name alone.

## 5. Rust Target-Feature Safety

A function annotated with `#[target_feature(enable = "feature")]` may only execute when the running CPU/platform supports that feature. Calling it without proof is undefined behavior.

Use one of these proofs:

1. The feature is part of the documented compile-time target baseline.
2. The caller itself has all required `#[target_feature]` annotations.
3. A runtime detection macro succeeded immediately before the unsafe call.
4. A safe library token such as `fearless_simd::Level` proves availability.

A safe wrapper pattern:

```rust
pub fn add(out: &mut [f32], a: &[f32], b: &[f32]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::arch::is_x86_feature_detected!("avx") {
        // SAFETY: detection proves AVX; slice checks prove equal lengths.
        unsafe { add_avx(out, a, b) };
        return;
    }

    add_scalar(out, a, b);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
unsafe fn add_avx(out: &mut [f32], a: &[f32], b: &[f32]) {
    // Vector loop and scalar tail.
}
```

Do not combine `#[inline(always)]` with `#[target_feature]`. Rust disallows that combination. Ordinary `#[inline]` is a hint, not a requirement.

## 6. Compile-Time Versus Runtime Selection

### Compile-time selection

Use when every deployment CPU is known:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

or a named CPU:

```bash
RUSTFLAGS="-C target-cpu=znver4" cargo build --release
```

This can let LLVM optimize the whole crate for the baseline, but the binary may fail or execute undefined behavior on older hardware if it reaches unsupported instructions. It is unsuitable for broadly distributed binaries unless the installer/runtime enforces compatibility.

### Runtime multiversioning

Use a generic binary and isolate optional features in functions. Detection occurs in a safe wrapper. This preserves compatibility while exploiting newer CPUs.

For frequently called tiny kernels, select a function pointer or library token once at initialization or an outer processing boundary. Measure whether caching matters before adding complexity.

## 7. Instruction Mapping

The following table is a selection guide, not a promise about exact code generation. Intrinsic availability and signatures must be checked in the current Rust documentation.

| Operation | SSE / SSE2 (128-bit) | AVX / AVX2 (256-bit) | AArch64 NEON (128-bit) | Notes |
|---|---|---|---|---|
| Load `f32` | `_mm_loadu_ps` (SSE) | `_mm256_loadu_ps` (AVX) | `vld1q_f32` | Unaligned variants still require all bytes in bounds. |
| Store `f32` | `_mm_storeu_ps` | `_mm256_storeu_ps` | `vst1q_f32` | Do not use streaming stores without a measured, correctly fenced design. |
| Add `f32` | `_mm_add_ps` (SSE) | `_mm256_add_ps` (AVX) | `vaddq_f32` | Element-wise order is preserved. |
| Multiply `f32` | `_mm_mul_ps` (SSE) | `_mm256_mul_ps` (AVX) | `vmulq_f32` | FMA is a separate feature/semantic decision. |
| Add `i32` | `_mm_add_epi32` (SSE2) | `_mm256_add_epi32` (**AVX2**) | `vaddq_s32` | AVX alone does not provide the 256-bit integer form. |
| Add `u32` | `_mm_add_epi32` (SSE2) | `_mm256_add_epi32` (**AVX2**) | `vaddq_u32` | Hardware addition is bitwise identical for signed/unsigned wrapping add. |
| Saturating add `u8` | `_mm_adds_epu8` (SSE2) | `_mm256_adds_epu8` (AVX2) | `vqaddq_u8` | Match the scalar oracle with `saturating_add`. |
| Compare equal `u8` | `_mm_cmpeq_epi8` (SSE2) | `_mm256_cmpeq_epi8` (AVX2) | `vceqq_u8` | Convert masks with the intended API, not transmute. |
| Compact byte mask | `_mm_movemask_epi8` (SSE2) | `_mm256_movemask_epi8` (AVX2) | No direct equivalent | NEON often uses reductions, narrowing, table logic, or a portable library mask operation. |
| Sum absolute byte differences | `_mm_sad_epu8` (SSE2) | `_mm256_sad_epu8` (AVX2) | `vabdq_u8` plus widening/pairwise reduction | Useful for image metrics and match scoring. |
| Widen `u8` to `u16` | unpack with zero: `_mm_unpacklo_epi8`, `_mm_unpackhi_epi8` | AVX2 unpack or conversion intrinsics | `vmovl_u8` / widening pairwise operations | Widen before accumulating enough values to avoid overflow. |
| Select by mask | AND/ANDNOT/OR; some later blends | `_mm256_blendv_ps` for float masks; logical select for integers | `vbslq_*` | Define NaN and signed-zero behavior when selection is replacing min/max. |
| Interleaved RGB load | unpack/shuffle sequence | AVX2 shuffle/unpack sequence | `vld3q_u8` | NEON structured loads can naturally deinterleave RGB. SSE2-only versions can be shuffle-heavy. |

## 8. SSE (SSE1) Versus SSE2

### SSE / SSE1

Use SSE for the 128-bit packed `f32` operations covered here:

- Four-lane `f32` add, subtract, multiply, divide.
- Floating comparisons and bitwise masks.
- Shuffles and scalar/packed conversions.
- Unaligned 16-byte loads and stores.

Avoid MMX-based integer designs. SSE2 XMM integer operations are generally the maintainable baseline for integer SIMD in modern Rust code.

### SSE2

SSE2 adds:

- Packed integer arithmetic in XMM registers.
- Packed `f64` arithmetic.
- Byte/word saturation.
- Integer shifts, comparisons, packing/unpacking.
- `_mm_movemask_epi8` for compacting byte sign bits.
- `_mm_sad_epu8` for byte SAD.

Remember that some operations familiar from later x86 generations are not SSE2:

- Horizontal add is SSE3.
- Byte shuffle (`pshufb`) is SSSE3.
- Integer min/max coverage expands in SSE4.1.
- CRC32 is SSE4.2.

Do not accidentally raise the minimum ISA by selecting these intrinsics in an SSE2 path.

## 9. AVX Versus AVX2

### AVX

AVX provides 256-bit packed floating-point arithmetic and VEX forms of many 128-bit operations. It is appropriate for `f32`/`f64` map-style kernels when memory and dependencies permit.

Do not use AVX detection as proof for AVX2 intrinsics. For example:

- `_mm256_add_ps` requires AVX.
- `_mm256_add_epi32` requires AVX2.

### AVX2

AVX2 extends most integer operations to 256 bits and adds gathers and richer integer operations. It is the normal 256-bit x86 choice for byte scanning, integer transforms, compression matchers, and image kernels.

Several AVX2 shuffle/unpack instructions still operate independently within 128-bit halves. Verify lane-crossing behavior rather than assuming a 256-bit permutation is global.

### AVX/SSE boundaries

Keep AVX kernels isolated in target-feature functions and let the compiler manage ABI transitions. Avoid legacy SSE inline assembly inside AVX code. Inspect assembly when a hot call boundary mixes AVX and non-AVX code; do not add `vzeroupper` manually without understanding the ABI and measuring the result.

## 10. NEON / AdvSIMD

NEON provides 64- and 128-bit vector operations. AArch64 exposes 32 128-bit SIMD/floating-point registers. Important NEON strengths include:

- Widening and narrowing arithmetic.
- Saturating arithmetic and saturating narrowing.
- Pairwise add and add-long operations.
- Structured loads/stores (`vld2`, `vld3`, `vld4`) for interleaved data.
- Table lookup/shuffle operations.
- Efficient multiply-accumulate patterns.

NEON naming conventions help read intrinsics:

- `q` usually denotes a 128-bit vector form, such as `vaddq_u32`.
- Suffixes encode lane type, such as `_u8`, `_s16`, `_f32`.
- `l` often indicates widening/long operations.
- `n` often indicates a scalar immediate/broadcast operand.
- `lane` variants operate with a selected lane.

Optional AArch64 extensions are not implied by baseline NEON. Examples include dot-product instructions and BF16. Detect or compile-gate them separately.

## 11. Loads, Stores, and Alignment

### x86 aligned loads

- `_mm_load_ps` requires 16-byte alignment.
- `_mm256_load_ps` requires 32-byte alignment.
- Violating the documented alignment can fault and is invalid Rust code.

### x86 unaligned loads

- `_mm_loadu_ps` and `_mm256_loadu_ps` do not impose those alignment requirements.
- They still require the complete loaded range to be valid and initialized.
- Modern x86 often handles unaligned data efficiently when the access does not cross costly boundaries, but benchmark the real workload.

### NEON loads

Use `vld1q_*`/`vst1q_*` for contiguous vectors. Do not overstate alignment guarantees: the pointer must be valid for the full access and meet Rust's object validity rules. Structured loads may be beneficial when the memory is genuinely interleaved.

### Rust allocation alignment

A `Vec<T>` guarantees alignment for `T`, not arbitrary 16/32/64-byte over-alignment. Do not use aligned x86 loads on ordinary `Vec<f32>` merely because heap addresses often look aligned. Use an over-aligned wrapper/allocator or unaligned loads.

## 12. Tails

Preferred strategies in order:

1. **Scalar epilogue:** simplest, portable, and usually best for short tails.
2. **Library masked load/store:** useful when the API supports it efficiently and the benchmark wins.
3. **Padded input owned by the API:** only when padding is initialized and part of the contract.
4. **Overlap the final full vector:** only for operations where duplicate processing is safe, all accesses remain in bounds, length is at least one vector, aliasing is controlled, and repeated stores are observationally equivalent.

Never implement a tail by blindly loading one full vector past the logical end.

## 13. Register and Lane Pressure

More lanes increase work per instruction but can also increase:

- Register pressure and spills.
- Shuffle complexity.
- Tail cost for small inputs.
- Code size from multiversioning.
- Downstream latency before a result can be reduced.
- Memory bandwidth demand.

For reductions, use independent accumulators only when the dependency chain limits throughput. For map kernels, avoid retaining more vectors than necessary.

## 14. Common Architecture Mistakes

- Detecting `avx` and then calling AVX2 integer intrinsics.
- Assuming x86 aligned loads are safe on `Vec<T>`.
- Putting feature detection inside the chunk loop.
- Using an SSE3/SSSE3/SSE4 intrinsic in an SSE2 function.
- Assuming NEON has `_mm_movemask_epi8` semantics.
- Ignoring 128-bit sub-lanes in AVX2 shuffles.
- Assuming AArch64 NEON implies `dotprod` or BF16.
- Compiling the whole binary with `target-cpu=native` and distributing it.
- Using a target-feature function through an unchecked function pointer.
- Treating intrinsic names as guaranteed single instructions.
