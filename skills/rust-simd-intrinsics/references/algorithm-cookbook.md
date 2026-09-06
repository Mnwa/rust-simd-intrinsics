# SIMD Algorithm Cookbook

Last verified: 2026-08-23.

Each recipe starts with a semantic contract and then describes a vector plan. Treat the instruction mappings as candidate implementations, not performance guarantees. Check the exact intrinsic documentation and emitted assembly for the selected target.

## 1. Contiguous Unary and Binary Maps

Examples: scale, bias, add, XOR, threshold, byte translation.

### Scalar oracle

```rust
pub fn affine(out: &mut [f32], input: &[f32], scale: f32, bias: f32) {
    assert_eq!(out.len(), input.len());
    for (dst, &x) in out.iter_mut().zip(input) {
        *dst = x * scale + bias;
    }
}
```

### Vector plan

1. Broadcast constants once.
2. Load one or more contiguous vectors.
3. Apply lane-wise operations.
4. Store.
5. Finish with a scalar tail.

### Candidate mappings

| Operation | SSE | SSE2 | AVX | AVX2 | NEON |
|---|---|---|---|---|---|
| `f32` load/store | `_mm_loadu_ps` / `_mm_storeu_ps` | same | `_mm256_loadu_ps` / `_mm256_storeu_ps` | same | `vld1q_f32` / `vst1q_f32` |
| `f32` add/mul | `_mm_add_ps`, `_mm_mul_ps` | same | `_mm256_add_ps`, `_mm256_mul_ps` | same | `vaddq_f32`, `vmulq_f32` |
| `u32` wrapping add | — | `_mm_add_epi32` | requires 128-bit integer path | `_mm256_add_epi32` | `vaddq_u32` |

Fused multiply-add is not part of baseline SSE/SSE2/AVX/AVX2. FMA is a separately enabled x86 feature. Use it only when the semantic contract permits one-rounding behavior and runtime support is proven.

### Best practices

- Auto-vectorization should handle simple maps first.
- Keep input/output length checks outside the loop.
- Avoid a callback or trait-object call per element.
- Do not use approximate reciprocal or reciprocal-square-root without an error contract.

## 2. Reductions and Dot Products

Examples: sum, energy, dot product, checksum accumulator.

### Contract decision

Choose one:

- **Strict order:** result must match a scalar left fold. Parallel/vector reassociation is not allowed for floating point.
- **Numerically bounded:** a stated tolerance or ULP bound is acceptable.
- **Wrapping integer:** arithmetic is modulo `2^N`.
- **Widening integer:** lanes accumulate into a larger type to avoid overflow.

### Vector plan

- Maintain two to four independent vector accumulators if latency limits throughput.
- Reduce vector accumulators after the main loop.
- Accumulate the scalar tail.
- For float dot products, decide whether FMA is allowed.

### Pitfalls

- Horizontal reduction changes floating-point association.
- A `u8` accumulator overflows almost immediately; widen before accumulation.
- A vector reduction may be slower for tiny slices.
- Multiple accumulators can increase register pressure and spills.

### Widening byte sum

A robust hierarchy for summing `u8`:

1. Widen `u8` to `u16` or use pairwise add-long.
2. Periodically widen/accumulate into `u32`.
3. Reduce to `u64` if the total length can exceed `u32::MAX / 255`.

Candidate instructions:

- SSE2: unpack bytes with zero, add words/dwords, or use `_mm_sad_epu8(x, zero)` to produce two 64-bit partial sums.
- AVX2: `_mm256_sad_epu8` or unpack/add paths; remember 128-bit lane structure for some operations.
- NEON: `vpaddlq_u8`, then pairwise/widening accumulation such as `vpadalq_*` where available.

## 3. Sum of Absolute Differences (SAD)

Useful for image blocks, compression matching, and similarity metrics.

### Scalar oracle

```rust
pub fn sad_u8(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&x, &y)| x.abs_diff(y) as u64)
        .sum()
}
```

### x86 plan

- SSE2 `_mm_sad_epu8` computes absolute byte differences and partial sums.
- AVX2 `_mm256_sad_epu8` processes 32 bytes and returns multiple 64-bit partial sums.
- Accumulate partial sums in sufficiently wide lanes.

### NEON plan

- `vabdq_u8` computes absolute byte differences.
- Widen/pairwise-add with `vpaddlq_u8` or accumulate-long operations.

### Pitfalls

- Verify output lane layout before reducing.
- Ensure total accumulation cannot overflow.
- For early-exit matching, full SAD may waste work; compare a running bound in chunks.

## 4. Byte Equality, `memchr`, and Delimiter Scanning

Goal: find the first byte equal to a needle.

### Scalar oracle

```rust
pub fn find_byte(bytes: &[u8], needle: u8) -> Option<usize> {
    bytes.iter().position(|&x| x == needle)
}
```

### x86 vector plan

1. Broadcast `needle`.
2. Compare bytes for equality.
3. Extract one bit per lane with movemask.
4. If the mask is nonzero, use `trailing_zeros` to locate the first lane.
5. Process the tail.

Candidate intrinsics:

- SSE2: `_mm_cmpeq_epi8`, `_mm_movemask_epi8`.
- AVX2: `_mm256_cmpeq_epi8`, `_mm256_movemask_epi8`.

### NEON vector plan

NEON has byte comparison but no exact baseline equivalent of x86 `pmovmskb`. Options include:

- Reduce whether any lane matched, then inspect/store the matching block.
- Apply weighted narrowing/shift strategies to create a bit mask.
- Use a library abstraction that supplies `to_bitmask` and inspect its lowering.

Do not transliterate the x86 algorithm without accounting for mask extraction cost.

### Multi-needle scan

For two or four delimiter bytes, compare against each broadcast value and OR the masks before extraction. This often beats scalar branches.

### Pitfalls

- The first set bit maps to byte order; test both little- and big-endian targets if manually constructing masks.
- A vector path can lose for very short haystacks. Consider a size threshold only after measuring.
- Do not overread the slice to avoid a tail.

## 5. First Mismatch and Longest Equal Prefix

Useful in compression matchers and byte-string comparisons.

### Scalar oracle

```rust
pub fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let limit = a.len().min(b.len());
    a[..limit]
        .iter()
        .zip(&b[..limit])
        .position(|(&x, &y)| x != y)
        .unwrap_or(limit)
}
```

### x86 plan

- Compare one vector from each input.
- Extract equality bits.
- If all lanes match, advance.
- Otherwise invert/mask the equality bits and use `trailing_zeros` for the mismatch lane.

For SSE2, all 16 equality bits set gives `0xffff`; for AVX2, all 32 set gives `u32::MAX` after conversion. Avoid sign-extension mistakes from the intrinsic's signed integer return type.

### NEON plan

- Compare bytes.
- Reduce to detect an all-equal block.
- On the first mismatching block, use a bitmask helper, store-and-scan fallback, or a narrowing/index strategy.

The best NEON mask extraction depends on the larger matcher. Benchmark it independently; it is often the dominant cost.

### Compression-specific practices

- Check cheap candidate tags/hashes before loading a full match block.
- Keep the maximum match length outside inner loops.
- Avoid repeated slice bounds checks by taking equal-length bounded prefixes once.
- Benchmark random, repetitive, and mixed Unicode/binary data separately.
- Do not read past the logical limit even if the allocation has spare capacity.

## 6. Saturating Arithmetic

Examples: image brighten, audio mixing, byte-domain scoring.

### Scalar oracle

```rust
pub fn add_saturating_u8(out: &mut [u8], a: &[u8], b: &[u8]) {
    assert_eq!(out.len(), a.len());
    assert_eq!(out.len(), b.len());
    for ((dst, &x), &y) in out.iter_mut().zip(a).zip(b) {
        *dst = x.saturating_add(y);
    }
}
```

Candidate instructions:

- SSE2: `_mm_adds_epu8`.
- AVX2: `_mm256_adds_epu8`.
- NEON: `vqaddq_u8`.

Keep saturating and wrapping operations distinct in naming and tests. They are not interchangeable optimizations.

## 7. Branchless Select, Clamp, and Classification

### Vector plan

1. Compute a comparison mask.
2. Select lane values from `if_true` and `if_false`.
3. Store.

Mappings:

- SSE/SSE2 without blend: bitwise `(mask & true) | (!mask & false)` using correctly typed masks.
- AVX blend instructions require the corresponding feature and often use sign bits or immediate masks; verify semantics.
- NEON: `vbslq_*` selects bits according to the mask.
- `wide`, `fearless_simd`, and `std::simd`: prefer their mask/select APIs.

### Float clamp warning

Scalar `x.max(lo).min(hi)`, x86 min/max instructions, Rust methods, and library `fast_min`/`fast_max` can differ for NaNs and signed zero. Specify behavior before choosing an instruction.

### When branchless loses

- The branch is highly predictable.
- Only a rare branch performs expensive work.
- Both branches compute costly values eagerly.
- Mask extraction or blending costs more than the branch.

## 8. ASCII Validation and Range Tests

Goal: determine whether all bytes are ASCII (`< 128`) or lie in a range.

### ASCII strategies

- x86 SSE2/AVX2: inspect high bits with movemask; ASCII is valid when the mask is zero.
- NEON: compare against `0x7f` and reduce, or use a maximum reduction when available at the required feature level.
- Portable APIs: compare against a splat and call mask `.all()`.

### Signed comparison trap on x86

SSE2 lacks every unsigned comparison form. For unsigned byte ranges, common strategies include:

- XOR each byte with `0x80` and use signed comparison.
- Use saturating subtraction and equality/range logic.
- Split the test into high-bit and lower-range checks.

Prove the transformation with exhaustive tests across all 256 byte values.

## 9. Widen, Multiply, Accumulate, Narrow

Common in audio, image, and fixed-point code.

### Pipeline

1. Load narrow integers.
2. Widen signed or unsigned lanes correctly.
3. Multiply/accumulate in wider lanes.
4. Apply rounding if required.
5. Saturate or wrap according to the contract.
6. Narrow and store.

### Architecture strengths

- SSE2: unpack low/high halves, multiply 16-bit lanes, pack with signed/unsigned saturation.
- AVX2: wider equivalents, but shuffles/unpacks often operate independently in 128-bit halves.
- NEON: widening operations (`vmovl` family), multiply-long, saturating rounding/narrowing instructions.

### Pitfalls

- Sign extension versus zero extension.
- Intermediate overflow before a later saturation.
- Different rounding on narrowing shifts.
- AVX2 lane-local ordering surprises.
- Accidentally changing fixed-point scale.

Use exhaustive tests for small lane types and boundary-focused tests for wider types.

## 10. Interleaved RGB/RGBA Data

### Prefer data layout first

If repeated operations target one channel at a time, a structure-of-arrays layout can eliminate deinterleaving. Changing layout may yield more than changing instructions.

### NEON structured load plan

NEON provides `vld3q_u8` and `vld4q_u8` families that load interleaved RGB/RGBA data into separate vectors. This is a natural fit for image kernels.

### x86 plan

x86 usually requires shuffle/unpack sequences. Candidate strategies differ by SSE2, SSSE3, and AVX2:

- SSE2: unpack and shift logic, often instruction-heavy.
- SSSE3: byte shuffle can simplify deinterleaving, but SSSE3 is not SSE2.
- AVX2: `vpshufb` remains lane-local to 128-bit halves and needs careful recombination.

Do not place an SSSE3 shuffle inside an SSE2-labeled kernel.

### Alternative

Process several pixels in an array-of-structures form if the arithmetic naturally uses all channels. Avoid deinterleaving merely by habit.

## 11. Lookup Tables and Byte Shuffles

Examples: nibble transforms, classification, Base64/hex helpers.

- x86 SSSE3 `pshufb`/AVX2 `vpshufb` is powerful, but SSSE3 is outside this skill's SSE2 baseline and must be dispatched separately.
- NEON table lookup (`vtbl`/`vqtbl` families) differs in table size and supported architecture version.
- Portable shuffles can scalarize or expand; inspect code generation.

For a 256-entry table indexed by arbitrary bytes, a direct vector gather is usually unavailable or expensive. Split by nibbles only when the transform can be decomposed correctly.

## 12. Early Exit and Chunk Bounds

SIMD is most effective when many full chunks are processed. For algorithms with a cutoff:

- Accumulate per chunk and test at a coarse boundary.
- Check a cheap scalar prefix for very small limits.
- Avoid reducing every vector if the bound can be tested less often safely.
- Preserve the exact point at which the scalar contract says to stop when the return value depends on order.

An early-exit vector algorithm that reports only “some lane failed” must recover the first failing lane before returning an index.

## 13. Recipe Selection Checklist

For every implementation, record:

- Scalar contract.
- Lane type and vector width.
- Required CPU feature.
- Load/store validity proof.
- Tail strategy.
- Overflow and float behavior.
- Mask extraction strategy.
- Expected bottleneck.
- Assembly evidence.
- Benchmark evidence on each deployment CPU family.
