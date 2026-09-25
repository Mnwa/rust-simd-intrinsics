# Reductions: semantics, construction, and placement

See [API capabilities](api-capabilities.md) and sources [R1–R6] in
[the source index](vectorization-sources.md). A lane-wise `min` is not a horizontal
minimum. A horizontal reducer consumes one register; an array reduction also
needs a loop, an overflow budget, an empty-input policy and a tail.

## Write the contract first

| Operation | Empty/identity | Important distinction |
|---|---|---|
| wrapping integer sum | 0 | equivalent modulo 2^w; not checked sum |
| wrapping integer product | 1 | a zero-filled tail is wrong |
| AND | all ones | use the element width, not an untyped sentinel |
| OR / XOR | 0 | masking must preserve identity |
| any / all | false / true | use the library's mask reduction |
| min / max | preferably `Option<T>` for a slice | identity alone loses empty/all-NaN information |
| argmin / argmax | `None` | carry index and validity; define first/last tie |
| float sum/product | specified by the caller | reassociation and FMA may change results |

Checked and signed saturating addition are not interchangeable with wrapping
addition. For example, signed saturating sums can change when regrouped. Do not
justify FP reassociation using mathematical associativity.

For FP min/max, specify whether NaNs are propagated or ignored, what all-NaN
returns, and the treatment of +0/-0. `std::simd` min/max reductions ignore NaNs
unless all lanes are NaN, and may return either sign of equal zero [R1]. A tree
using arbitrary hardware min/max instructions is not automatically equivalent.

## Missing reducer: comparable candidates, not one dogma

1. Use the actual built-in reducer when it has the required semantics.
2. Compose fixed-width shuffle/combine operations. For four wrapping u32 lanes:
   `p = v + rotate(v, 2); total = p + rotate(p, 1); result = total[0]`.
   Replace both additions with multiplication for the wrapping product.
3. Accumulate in vectors, then fold the final lane array once. This is a valid
   implementation candidate; inspect whether the compiler keeps it in registers.
4. Consider a small ISA helper behind existing dispatch. The rest of the kernel
   should stay portable if only the reducer needs specialization.

The complete Fearless implementations are in `examples/vectorization-v2/src/fearless.rs`. They use v1 built-in numeric reducers, composed bitwise trees, and native-width accumulation with
1/2/4 independent accumulators. The generic accumulator count is a normal const
parameter; no unsupported expression such as `::<{V::LEN / 2}>` is required.

The raw SSE2 and NEON widened-byte implementations in `arch.rs` are deliberately
small specializations. The SSE2 path uses SAD against zero, not a wrapping u8
accumulator. All public raw-intrinsic wrappers check CPU support where necessary.

## Reduce at the right frequency

A long array normally has a vertical phase and one horizontal epilogue:

```text
acc0, acc1, ... = identity
for full group:
    acc0 = op(acc0, load(block0))
    acc1 = op(acc1, load(block1))
acc = combine(acc0, acc1, ...)
answer = horizontal_reduce(acc)
answer = scalar_tail(answer)
```

Measure 1, 2 and 4 accumulators. More accumulators can shorten dependency chains
but also increase register pressure and code size [R7]. Some wide shuffles do not
cross 128-bit sub-blocks; a complete wide reduction must combine those blocks.
Never implement a 256/512-bit reducer by repeating only a 128-bit-local shuffle.

Exceptions: one output per block, a required early exit, or an overflow budget may
require periodic reduction. Short-input kernels can be dominated by dispatch and
epilogue costs, so benchmark them separately.

## Derive the widening schedule

If one iteration adds at most M to an unsigned w-bit accumulator, it can accept
at most floor((2^w-1)/M) such increments starting from zero. Account for *every*
update to a lane and any final combine. u16 holds 257 additions of 255 exactly;
258 do not fit. Widen before overflow, not after it.

`sum_u8_widened` in the examples returns a u64 sum modulo 2^64 (also the exact sum
whenever it fits). That explicit contract avoids target-pointer-width assumptions.
`hamming` uses u64 for the same reason. A widened dot product must widen before
multiplication if the narrow product itself can overflow.

## Argmin/argmax

Define comparison on `(value, index)` lexicographically: lower value first, then
lower index for a first-occurrence argmin. Maintain a validity mask if padding can
have a legitimate sentinel value. The executable nightly examples instead process
only full vectors and handle the remainder scalar, so padded lanes cannot win.
The example stores indices in usize lanes, not truncated u32 lanes.

## Tests and evidence

Use lengths 0..=257, several large lengths, and offsets 0..=31. Include all zeros,
all maximum integers, alternating extremes, deterministic random values, product
inputs with and without zero, and ties in different vector blocks. Test the actual
vector body, not only a seven-element tail on an eight/sixteen-lane backend.

For floats test NaNs, infinities, both zero signs, subnormals and cancellation
under the declared contract. Tolerance checks are not a substitute for bitwise
comparison when an ordered result is required. Validate the oracle too.
