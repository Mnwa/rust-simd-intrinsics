# Executable vectorization recipes

Independent workspace; it does not update the original examples or their
libraries. Exact top-level pins: Fearless 0.7.0, wide 1.7.0. Portable SIMD is an
optional nightly feature. This patch contains no fabricated Cargo.lock.

```bash
# First connected run; only creates missing locks. Review and commit the lock.
python3 scripts/verify-vectorization.py --strict --scope all --bootstrap-lock
# Subsequent verification must reuse the recorded resolution.
python3 scripts/verify-vectorization.py --strict --scope all
# Optional style check once rustfmt is installed:
cargo +stable fmt --manifest-path examples/vectorization-v2/Cargo.toml
```

Run commands from the repository root. Stable-only or nightly-only scopes are
explicitly selectable. `--scope stable` is not a claim that nightly was tested.
The workflow runs both scopes on x86-64 and ARM64 hosts; its presence does not mean
those jobs have already executed. Pin the validated nightly date for release
reproducibility and record it via `--nightly-toolchain nightly-YYYY-MM-DD`.

## Contents and contracts

| Module | Examples | Important contract |
|---|---|---|
| scalar | all oracles, private histogram | optimized scalar-written baselines; no disabled autovec |
| arch | u32 sum, widened byte sum, prefix scan, byte search | checked SSE2/AVX2/NEON; no overread |
| fearless | fixed reducer trees, native 1/2/4 accumulators, four outputs, histogram score | modulo u32; scalar epilogue allowed |
| wide_examples | f32x8 sum and product | explicitly reassociated FP |
| portable | min/index reductions, scan, masks, filter, Hamming, hex, ASCII, sort4, Adler, stencil, statistics | nightly, explicit tails and oracle comparisons |
| math | bounded u32 log2 and histogram score | approximate score, not exact-sign replacement |

`arch`'s AVX2 scan and widened-byte sum intentionally use the SSE2 implementation.
NEON byte search intentionally uses the scalar implementation. These are labeled
fallbacks, not a claim that every function has every ISA variant. Tests enumerate
raw supported backends and Fearless's safely extracted available levels.

`tests/guard_pages.rs` places the end of Linux slices immediately before a protected
page. Other tests cover offsets, lengths 0..257, longer arrays, overflow, ties,
mask distributions and math special cases. These source files are checks to run;
see `VECTORIZE-V2-STATUS.md` for what was actually executed while preparing the patch.

## Codegen and timing

```bash
python3 scripts/emit-vectorization-asm.py generic --features fearless,wide
python3 scripts/emit-vectorization-asm.py native --features fearless,wide
cargo +stable run --release --locked --manifest-path examples/vectorization-v2/Cargo.toml \
  --features fearless,wide --bin recipe-bench -- 65536 7 20 > samples.csv
cargo +nightly run --release --locked --manifest-path examples/vectorization-v2/Cargo.toml \
  --features portable,fearless,wide --bin recipe-bench -- 65536 7 20 > samples-nightly.csv
```

Parameters are element count, sample count and minimum calibration duration in
milliseconds. Repeat over short and long sizes, e.g. 0, 1, 3, 7, 15, 31, 257, 4096,
65536 and 1048576. Inputs/LUT are prebuilt; result allocation is included. The
runner emits observations, not automatic claims of statistical significance.
Public raw wrappers include feature/support checks. Fearless selected-level cases
exclude `Level::new()` but still execute the dispatch context. Report this distinction.

For controlled generic/native builds the codegen script explicitly sets CPU and
cleans inherited rustflags, stages outside repository config ancestry and records
Cargo-home config hashes. Benchmark builds must use the same documented CPU flags
for baseline and candidate; do not compare a native candidate against a generic
baseline by accident. No timings from this patch-preparation environment exist.
