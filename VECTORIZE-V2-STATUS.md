# Vectorization v2 — delivery and validation status

Prepared on 2026-09-06 for `Mnwa/rust-simd-intrinsics`.

## Read this before applying

This delivery is an **additive Git patch plus a checkout-aware integration tool**.
It is not a diff generated against a verified upstream commit. Retrieval of the
GitHub repository failed in the authoring environment, so the upstream revision,
current file contents, and applicability to its current HEAD are unknown.
`git apply --check` must be run in the real checkout.

The patch adds independent new paths. Then
`scripts/integrate-vectorization-v2.py` reads the actual checkout and generates a
normal integration diff, or applies it with `--write`. It preserves the original
SKILL frontmatter and body, adds routing/supplements, updates the verification and
assembly entry points, and ignores generated reports. It does not invent a base
commit or use guessed deletion hunks. Existing files that would be changed must
be clean for `--write`; symlink targets are refused.

See [application instructions](APPLY-VECTORIZATION-V2.md).

## Implemented in this delivery

- Eight detailed reference documents: workflow/routing, API capabilities,
  reductions, transformation patterns, lookup/vector math, real upstream case
  studies, primary sources, and validation requirements.
- Independent Rust workspace with scalar oracles, raw SSE2/AVX2/NEON helpers,
  optional Fearless 0.7.0 and wide 1.7.0 candidates, and nightly portable SIMD.
- Missing numeric reducers, one/two/four accumulators, widening, argmin/argmax,
  prefix scan, byte masks/search/filter, Hamming distance, ASCII/hex,
  private histograms, bounded Adler-32, small sorting network, output-lane dot
  products, a stencil, fused statistics, and approximate histogram scoring.
- Differential/API/backend/overflow/tail tests and Linux guard-page tests.
  The separate Fearless doubling test includes full vector blocks and overflow;
  **the unavailable original `examples/src/fearless_impl.rs` is not modified or
  directly tested by this new test**. The verifier's `--legacy` option runs the
  original workspace when it is present in the actual checkout.
- Strict stable/nightly verification, isolated codegen capture, a sample-based
  benchmark executable, CI matrix, eleven agent-evaluation tasks and a scorer.

The case studies describe techniques in real upstream implementations. The new
Rust examples are independent implementation candidates, not copied upstream
code and not locally established performance successes.

## Verification actually performed

| Check | Result | Scope |
|---|---|---|
| Python tooling unit tests | 14 passed | Integration, flags, missing tools, eval accounting |
| Independent Python algebra tests | 8 passed | Trees, scans, widening organization, sorting, Adler, SWAR, polynomial model, conflict counterexample |
| Python syntax; JSON/TOML/YAML parsing; eval schema | Passed | Structure of the delivered files, not Rust compilation |
| Additive patch and generated integration diff | Passed on a synthetic Git fixture | Mechanical application only; not upstream HEAD compatibility |
| Integration repeat and dirty-file refusal | Passed | No duplicate routing and no overwrite of local original-file edits |
| Git whitespace/error check | Passed | Added patch content |
| Rust compilation, API probes and runtime tests | **Not run** | Cargo/rustc are absent |
| Rust formatting | **Not run** | rustfmt is absent |
| Guard-page/ISA execution, assembly inspection | **Not run** | Rust toolchain unavailable |
| Benchmarks and performance gates | **Not run** | No measured speedups claimed |
| GitHub Actions and agent evaluation experiment | **Not run** | Definitions supplied; sample eval data are explicitly unmeasured |

Python model tests validate algebra in a separate implementation. They do not
prove that Rust compiles, that intrinsics lower as intended, or that backend code
is correct. The strict Rust verifier was invoked and correctly returned failure
with a missing-Cargo report; that expected failure is **not** a Rust test pass.

## Dependency resolution and follow-up validation

Top-level optional dependency versions are exact. A `Cargo.lock` is deliberately
not fabricated. Resolve dependencies in a connected environment using
`--bootstrap-lock`, inspect the actual lockfile, and commit it. Subsequent runs
use `--locked`; the bootstrap option never updates an existing lockfile.
Stable/nightly channels in the initial CI definition are bootstrap choices, not
claims of a previously verified compiler version. Record and pin the exact
successful toolchains after real execution.

Runtime coverage depends on the CPU. Enumerating available backend tokens never
means executing unsupported hardware. AVX2 dispatch deliberately reuses the SSE2
byte-sum/scan helper, and the raw NEON search path uses the scalar oracle; these
fallbacks are identified rather than presented as distinct implementations.

The `log2` recipe has an analytic series-truncation bound and sampled model tests,
but not a certified complete floating-point error bound. Approximate histogram
scores are not guaranteed to preserve a threshold decision near zero. Preserve
the scalar decision path when exact decision equivalence is required.
