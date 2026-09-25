# Version- and type-specific API capabilities

The recipe workspace pins `fearless_simd = "=1.0.0"` and `wide = "=1.7.0"`.
The original examples also use Fearless 1.0.0, with the optional macro crate
0.1.0 enabled by `fearless-example`; their `wide` pin remains 1.6.1.
Generate dependency locks with Cargo and reuse the reviewed resolution.
Use `tests/api.rs` as a compile probe, not generated API summaries as authority.

| Operation | nightly `std::simd` | `wide::f32x8` 1.7.0 | Fearless 1.0.0 |
|---|---|---|---|
| sum | `SimdFloat/SimdUint::reduce_sum` | `reduce_add` | `SimdBase::reduce_sum` |
| product | `reduce_product` | `reduce_mul` | `SimdBase::reduce_product` |
| min / max | `reduce_min` / `reduce_max` | check concrete type's rustdoc | `reduce_min/max`, plus `reduce_min/max_precise` |
| integer bit reductions | `reduce_and/or/xor` | check concrete integer type | helper |
| mask any/all | `Mask::any/all` | type-specific mask representation | `any_true/all_true` |
| scalar bitmask | `Mask::to_bitmask` | check concrete type | `to_bitmask` |

Sources: [R1–R6](vectorization-sources.md). A method on f32x8 does not establish
its existence on u8x16. Product is **not** named `reduce_mul` in portable SIMD.
Fearless 1.0 has numeric and mask reducers; only the integer bit reductions in
this table still need helpers. FP sum/product order is stable across backends for
a fixed vector type and lane count, not across native widths or slice algorithms.
See [v1 semantics and migration](libraries.md#fearless-v1-migration).
Do not rely on older auto-generated Context7 examples claiming `horizontal_add`
without checking the dependency source actually compiled by the project.

## Stable/nightly and dispatch

Portable SIMD uses `#![feature(portable_simd)]`; the example crate gates it behind
`portable`, and records the selected nightly toolchain in every verification
report. A pinned nightly can be passed explicitly; changing it is a tested update,
not an implicit assertion that all nightly releases share the same API.

Fearless's `dispatch!` enters a target-feature-aware context. The public examples
accept a `Level` variant for preselected-kernel measurements as well as a normal
`Level::new()` entry point. Never construct an unsupported token with `unsafe` to
force a test. `tests/fearless_backends.rs` enumerates safely extracted x86 tokens;
on other architectures it uses the detected level.

`wide` uses compile-time cfg branches in the dependency [R5]. Adding
`#[target_feature]` only to a caller does not rerun those cfg choices. A native
build may use a different representation from a generic build, but that is a
property of how the dependency was compiled. Inspect both.

## Capability verification checklist

Inspect Cargo.toml/Cargo.lock, resolve the concrete vector type and relevant
trait, compile a minimal probe, then inspect the codegen. Record version, target,
features and flags. A docs page for `latest` is a discovery aid, not a version pin.
If a method is absent, compare a composed primitive and a scalar epilogue before
moving the whole function to raw intrinsics.
