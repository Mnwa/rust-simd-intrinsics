# Lookup tables and vector mathematics

See [R13–R15](vectorization-sources.md) for production vector-math sources and the
Fearless mixed portable/ISA example. The executable histogram-score example is
an educational, explicitly approximate kernel, not a general-purpose libm port.

## Start from the complete expression

For `C - sum(c[i] * (depth[i] + log2(c[i])))`, define the contribution for c=0 as
zero. Evaluating `0 * log2(0)` is not an implementation of that definition. Use a
safe logarithm argument or an explicit zero path. Keep the same scalar tail.

`math.rs` contains a scalar reference and an original atanh-series approximation
for positive u32 counts. `portable::histogram_score` uses f64x4 polynomial
arithmetic; `fearless::histogram_score` supplies a stable f64x4 version. Both use
a caller-built 256-entry f64 table for small values. It explicitly
handles all-small, all-large and mixed blocks. The table is constructed outside
the measured kernel and must be included in end-to-end timing when not reused.

The f64 SIMD path uses scalar lane preprocessing and scalar table loads. That cost
is intentional and visible; do not describe it as hardware gather. Compare it
with a direct vector approximation and a real ISA gather where supported. The
example demonstrates an algorithmic choice, not that mixed paths are faster.

## `select` is not a lazy branch

For an ordinary call such as `mask.select(expensive(), cheap())`, both arguments
are evaluated before selection. Benchmark whole-block decisions (`all`, `any`),
unconditional computation and mixed-block repair. Arrange validity checks so no
invalid load, division, conversion or logarithm is evaluated before masking.

A fast all-small branch only helps when its branch frequency and predictability
repay the mask and control-flow cost. Include all-small, all-large, alternating,
sparse-large and zero-heavy distributions.

## Derive the approximation instead of inventing coefficients

For n>0, choose e=floor(log2(n)), m=n/2^e in [1,2), z=(m-1)/(m+1) in [0,1/3).
Then

```text
log2(n) = e + (2/ln(2)) * (z + z^3/3 + z^5/5 + ...)
```

The supplied implementation takes ten terms through z^19/19. Its real-arithmetic
truncation error is bounded by

```text
(2/ln(2)) * (1/3)^21 / (21 * (1 - 1/9)).
```

This is **not a bound on total floating-point error**, nor on the final score. It
excludes rounding, scalar log2/table accuracy, accumulation, subtraction and FMA
choices. The tests sample the u32 domain and compare errors, but sampling is not
an exhaustive proof. No coefficient was copied from an external implementation.

The supported domain is u32 counts. Zero is handled separately. Do not export the
polynomial as a full float log2: negative values, infinities, NaNs and subnormals
need a different contract and special-value handling.

## A sign decision needs an error budget

An approximation near `score == 0` can reverse `score >= 0`. A proven fast decision
would need a total bound E: decide only outside [-E,E], otherwise recompute using
the required reference semantics. This patch deliberately exposes an approximate
numeric score, not an allegedly exact threshold decision with an arbitrary E.
Until such a bound exists, use the reference for decisions requiring equivalence.

Tests cover c=0, 1, 255, 256, powers of two, neighbors of powers, u32::MAX, mixed
blocks and all tails. The benchmark separates LUT setup, preselected backend
kernel, and full public call where those distinctions apply.
