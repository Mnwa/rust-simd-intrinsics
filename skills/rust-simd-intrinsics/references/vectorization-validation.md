# Verification, provenance and performance acceptance

## Explicit states

`scripts/verify-vectorization.py` writes a JSON report and per-command logs. States
are passed, failed, skipped and unsupported. Missing Cargo/toolchains/locks are
failures in strict mode and visible skips in local mode. The report's overall
status is never passed when a required check was skipped. Strictness applies to
the declared stable/nightly/all scope, not to an untested architecture.

New recipes are an isolated workspace. `--legacy` additionally runs the original
example manifest using its available stable SIMD features and detected nightly
features. Existing lockfiles are never updated by bootstrap; missing ones require
explicit `--bootstrap-lock`. Review and commit the first resolved lock. Exact
transitive versions cannot be claimed before resolution.

Debug and release both run. Stable configurations are default, Fearless, wide and
Fearless+wide. Nightly configurations are portable and all three libraries. The
API tests are executable probes, not assumed compatibility. Current snippets were
not compiled in the patch-preparation sandbox: see the status report.

## Reach the vector body

Use lengths 0..=257, offsets 0..31, several larger arrays and lengths based on the
actual backend width. Seven values are not enough to test an eight-lane SIMD body.
Include maximal values, alternating extremes, odd product inputs, all-zero inputs,
first/last ties and mask-density variants. Force a supported backend only through
checked library facilities. Tests log which backend actually executes and identify
intentional per-operation fallbacks.

Linux guard-page tests complement logical-boundary tests. Cross-compiling for
ARM64 is not equivalent to running on ARM64. Unsupported machine features must
remain visible rather than replaced by invented feature tokens. A native x86 build
is not a valid way to test fallback deployment portability.

## Flags and compiler evidence

`emit-vectorization-asm.py` clears inherited `RUSTFLAGS`,
`CARGO_ENCODED_RUSTFLAGS`, target rustflag variables and wrapper environment
variables; sets an explicit encoded flag list and target; disables Cargo-config
compiler wrappers; and stages the workspace outside ancestor repository configs.
Cargo-home configuration hashes are recorded, not silently claimed absent. On
x86-64 generic mode uses `target-cpu=x86-64`, not `native`. Generic does not mean
"disable all SIMD": baseline SSE2 is still legitimate.

It emits library and driver assembly/LLVM IR plus `-C remark=all` diagnostics.
Inspect the actual hot loop, not just whether any vector instruction exists.
Record compiler version, features, effective flags, lock hash and hardware.
Sources for flags and diagnostics are [R7,R17,R18](vectorization-sources.md).

## Measure the right baseline

The primary baseline is the best correct scalar-written release implementation
with autovectorization enabled. A no-autovec build is diagnostic, not the only
convenient opponent. Match optimization/CPU/FMA policies across implementations.
Measure short calls (dispatch dominates), cache-resident blocks and memory-scale
inputs. Include application-level cost: allocation, packing, transposition,
lookup initialization, downstream processing and actual early-exit distributions.

Compare 1/2/4 accumulators and vector widths empirically. Preserve per-sample
results and report noise, regressions and unsuccessful candidates. A microbenchmark
or an upstream number does not prove a local end-to-end speedup. Treat an unchanged
baseline supported by evidence as an acceptable optimization outcome.

## Evaluate the agent

`evals/vectorization-v2` supplies a task catalog, result schema and scorer for a
controlled without/current/updated-skill comparison. It is not a completed LLM
study. The scorer keeps failed/unsafe/skipped attempts in pass-rate denominators
and reports performance only for measured correct attempts. Compare paired task,
configuration, target and seed sets rather than unmatched aggregate averages.
