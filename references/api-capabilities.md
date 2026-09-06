# Version- and type-specific API capabilities

The example workspace pins `fearless_simd = "=0.7.0"` and `wide = "=1.7.0"`.
They are isolated optional dependencies; the original example workspace is not
upgraded. A lockfile must be generated in a connected Rust environment and then
committed/reused. No unverified registry checksums are fabricated in this patch.
Use `tests/api.rs` as a compile probe, not generated API summaries as authority.

| Operation | nightly `std::simd` | `wide::f32x8` 1.7.0 | Fearless 0.7.0 |
|---|---|---|---|
| sum | `SimdFloat/SimdUint::reduce_sum` | `reduce_add` | helper |
| product | `reduce_product` | `reduce_mul` | helper |
| min / max | `reduce_min` / `reduce_max` | check concrete type's rustdoc | helper |
| integer bit reductions | `reduce_and/or/xor` | check concrete integer type | helper |
| mask any/all | `Mask::any/all` | type-specific mask representation | `any_true/all_true` |
| scalar bitmask | `Mask::to_bitmask` | check concrete type | `to_bitmask` |

Sources: [R1–R6](vectorization-sources.md). A method on f32x8 does not establish
its existence on u8x16. Product is **not** named `reduce_mul` in portable SIMD.
Fearless's missing numeric reducers do not mean its mask reducers are missing.
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
