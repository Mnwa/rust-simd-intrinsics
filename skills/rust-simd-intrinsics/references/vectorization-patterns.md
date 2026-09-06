# Transformations before instructions

Each pattern must retain an explicit scalar oracle, tails, tests and a performance
comparison. See `examples/vectorization-v2/README.md` for executable file names.
Hardware/compiler constraints are sourced in [R7–R15](vectorization-sources.md).

## Choose what one lane means

For `out[j] = sum_i x[i] * weights[i][j]`, two candidates are:

- Lanes hold consecutive i: fast contiguous input, but horizontal reduction per
  output and possibly strided weights.
- Lanes hold consecutive j: broadcast x[i], load a row of weights, accumulate
  independent outputs. No horizontal reduction; each output can keep its original
  addition order. `portable::batched_dot4` and `fearless::batched_dot4` do this.

Include packing/transposition costs unless the layout is already supplied by the
caller or legitimately reused. The same choice appears in FIR convolution,
stencils, distance calculations, matrix microkernels and batched queries.
`portable::convolve3` maps lanes to adjacent outputs and preserves the three-term
arithmetic order. Never assert that every strict FP computation is unvectorizable:
LLVM supports some ordered reductions, and independent outputs are another route.

## Prefix scan / delta decoding

For four elements use shifts of *elements*, not per-lane bit shifts:

```text
v = [a,b,c,d]
v += shift_right_elements(v, 1, zero)
v += shift_right_elements(v, 2, zero)
v += splat(carry_in)
carry_out = v[3]
```

This is correct for addition modulo 2^32. Full prefix sums are returned, unlike a
reduction. The SSE2 example uses byte shifts of 4 and 8 within one 128-bit register.
A wider implementation needs carry propagation across its 128-bit sub-blocks.
Compare one-pass, two-pass and cache-blocked approaches, including memory traffic.
The Algorithmica case study [R8] explains why an algebraic rewrite alone need not
win on large arrays.

## Compare -> mask -> only the required scalar result

A first-match search needs a nonzero check and the least set bit, not a scan of
all lanes. Count needs population count; any/all need boolean mask reductions.
The SSE2 byte search uses movemask and trailing_zeros. Portable examples provide
first-match, count, common prefix and ASCII uppercase classification.

Stable filtering can iterate selected mask bits in increasing order and write
only those elements. It is a hybrid SIMD-classification/scalar-compaction baseline,
not a claim to implement AVX-512 compress. Measure dense, sparse and alternating
masks before investing in shuffle-table or hardware-compress specializations.
Never write a full vector into capacity that contains only `popcount(mask)` slots.

## Conflict updates: histogram is not scatter-add

`gather(hist, indices); +1; scatter(...)` loses increments for repeated indices:
lanes with [5,5] can both read old and both write old+1. Tests must include all
identical bytes. The example `scalar::histogram_private` uses four private
histograms and merges them. Its update phase is intentionally scalar; this is a
conflict-avoidance/layout candidate, not a mislabeled vectorized scatter.

Compare scalar, private histograms plus vectorized merging, sorted/run-length
aggregation, or an ISA-specific conflict-aware approach. Account for the 4x table
footprint and initialization. SIMD processing of already-built bins is a different,
easier problem than constructing the histogram.

## Rewrite a recurrence: Adler-32

For block x[0..B], before modular reduction:

```text
s1_new = s1_old + sum(x[i])
s2_new = s2_old + B*s1_old + sum((B-i)*x[i])
```

The previous dependency becomes an ordinary sum plus a weighted sum. The nightly
example uses B=16, u32 lanes and reduces modulo 65521 after each block. With states
below 65521, the maximum new s2 is below
`65520 + 16*65520 + 255*(16*17/2)`, safely below u32::MAX. Longer batching needs a
fresh bound. zlib-ng [R12] is the real optimized case study; the educational
implementation is not a port and does not inherit its performance.

## Popcount and Hamming distance

XOR corresponding bytes, count bits, then widen/reduce. The portable example uses
a SWAR byte-popcount sequence so it does not assume an AVX-512 popcount instruction.
The scalar oracle uses `count_ones`. Inspect whether the compiler substitutes a
hardware instruction and whether reduction overhead dominates short inputs.

## LUT classification and small sorting networks

A nibble-table byte classifier can use a 16-entry shuffle table [R10,R11]. A
256-entry f64 lookup is not the same operation. Bounds and out-of-range shuffle
semantics are part of the contract. `portable::classify_hex` supplies a simple
compare/select baseline before LUT tuning.

A four-element sorting network uses compare-exchange stages with fixed partner
permutations. `portable::sort4` implements it for u32 with a scalar oracle. This
is not a general replacement for slice sorting: compare scalar insertion sort,
code size, batch packing and actual input lengths.

## Fuse only with evidence

Fusing sum, sum-of-squares, min and max can remove repeated loads and temporaries.
`portable::statistics_u32` specifies wrapping u64 sum/sum-of-squares and optional
extrema. Extra accumulators can spill; benchmark the fused implementation against
separate optimized passes. A fused kernel is a candidate, not an automatic win.
