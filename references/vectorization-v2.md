# Vectorization v2: executable transformations

This supplement extends, rather than replaces, the existing safety and ISA
references. Read this routing page before choosing a library. Examples live in
`examples/vectorization-v2`, an independent Cargo workspace; existing examples
and dependency versions are not rewritten.

## Required workflow

1. State the contract: empty input, integer overflow, FP order/FMA/NaN/zero,
   aliasing, lengths, padding, dispatch and deployment CPU. Keep a scalar oracle.
2. Identify the structure: map, reduction, scan, search, compaction, conflict
   update, stencil, classification, or recurrence. Propose at least two lane
   organizations when there is an independent output dimension.
3. Identify the obstacle in compiler remarks and codegen, not by guessing from
   source syntax. A loop-carried dependency can call for an algebraic rewrite.
4. Check the actual dependency/type API. Use a built-in primitive first. Missing
   `reduce_sum` is a reason to build one helper, not to abandon portable SIMD.
5. Produce a small correct candidate, including all tails. Compare array-lane
   epilogue, shuffle tree and local architecture specialization when relevant.
6. Differential-test each supported backend, in debug and release. Do not forge
   feature tokens or treat cross-compilation as execution.
7. Compare against optimized scalar-written code with autovectorization enabled.
   Measure dispatch separately, but also measure the public end-to-end call.
8. Report correctness, measured performance, codegen, and remaining uncertainty
   separately. Keeping the baseline is a valid outcome when SIMD loses.

## Read on demand

| Task | Read | Executable starting point |
|---|---|---|
| sum/product/min/max/argmin | [reductions](reductions.md) | `fearless.rs`, `arch.rs`, `portable.rs` |
| layout, scan, recurrence, filter | [patterns](vectorization-patterns.md) | `arch.rs`, `portable.rs`, `scalar.rs` |
| library selection or missing methods | [API matrix](api-capabilities.md) | `tests/api.rs` |
| LUT, polynomial, histogram score | [math](lookup-and-vector-math.md) | `math.rs`, `portable.rs` |
| realistic manual vectorization | [case studies](case-studies.md) | prefix scan, Adler-32, byte search |
| tests, toolchains, measurements | [validation](vectorization-validation.md) | `scripts/verify-vectorization.py` |
| evidence | [sources](vectorization-sources.md) | upstream sources, not copied code |
| does the skill improve the agent? | `evals/vectorization-v2/README.md` | tasks, schema, scorer |

## Do not introduce these regressions

- Do not extract every lane every iteration just to reconstruct a scalar loop.
  Extracting a few lanes once in an epilogue is a benchmarkable candidate.
- Do not perform horizontal reduction on every block of a long reduction unless
  the output contract or bounded accumulator actually requires it.
- Do not assume `select` makes its expensive argument lazy.
- Do not use gather/add/scatter as a general histogram update.
- Do not zero-pad a product or let invalid argmin lanes win a tie.
- Do not turn an ordered FP sum into a tree without permission.
- Do not assume the widest vector or the largest unroll factor wins.
- Do not claim an ISA from a caller's `#[target_feature]` changes a dependency's
  previously evaluated `cfg(target_feature)` branches.

The checked-in Rust is a set of implementation candidates, not a universal speed
claim. Consult the validation report accompanying a patch for what was actually
compiled and run. Never convert a skipped check into a pass.
