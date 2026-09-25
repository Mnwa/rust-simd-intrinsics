# Authoritative Source Index

Last verified: 2026-08-23.

Use primary documentation. Re-check current versions and signatures before producing code because nightly Rust and third-party SIMD crates evolve quickly.

## Agent Skills format

- Agent Skills specification: <https://agentskills.io/specification>
- Agent Skills best practices: <https://agentskills.io/skill-creation/best-practices>
- Agent Skills overview: <https://agentskills.io/home>

The package follows the required `SKILL.md` plus optional `references/`, `scripts/`, and `examples/` layout. The skill name is lowercase kebab case and matches its directory name.

## Rust language and compiler

- Rust Reference, code generation attributes and `target_feature`: <https://doc.rust-lang.org/reference/attributes/codegen.html#the-target_feature-attribute>
- Rust Reference, conditional compilation target features: <https://doc.rust-lang.org/reference/conditional-compilation.html#target_feature>
- Rust `std::arch` module: <https://doc.rust-lang.org/std/arch/index.html>
- Rust runtime feature detection: <https://doc.rust-lang.org/std/arch/macro.is_x86_feature_detected.html>
- Rust AArch64 runtime feature detection: <https://doc.rust-lang.org/std/arch/macro.is_aarch64_feature_detected.html>
- Rust AArch32 runtime feature detection (currently nightly-only): <https://doc.rust-lang.org/std/arch/macro.is_arm_feature_detected.html>
- Rust platform support: <https://doc.rust-lang.org/rustc/platform-support.html>
- rustc code generation options: <https://doc.rust-lang.org/rustc/codegen-options/index.html>
- Cargo profiles: <https://doc.rust-lang.org/cargo/reference/profiles.html>
- rustc command-line output types (`--emit`): <https://doc.rust-lang.org/rustc/command-line-arguments.html#--emit-specifies-the-types-of-output-files-to-generate>
- `std::hint::black_box`: <https://doc.rust-lang.org/std/hint/fn.black_box.html>

Key safety facts to re-check there:

- Calling a target-feature function without support is undefined behavior.
- Feature implications are documented by Rust rather than inferred from CPU marketing names.
- `#[inline(always)]` cannot be combined with `#[target_feature]`.

## LLVM auto-vectorization

- LLVM Auto-Vectorization documentation: <https://llvm.org/docs/Vectorizers.html>

LLVM documents the Loop Vectorizer and SLP Vectorizer, cost modeling, reductions, runtime pointer checks, and reasons a loop may remain scalar.

## Portable SIMD

- Current nightly `std::simd`: <https://doc.rust-lang.org/nightly/std/simd/index.html>
- `Simd` type: <https://doc.rust-lang.org/nightly/std/simd/struct.Simd.html>
- Portable SIMD tracking issue: <https://github.com/rust-lang/rust/issues/86656>
- Portable SIMD project repository: <https://github.com/rust-lang/portable-simd>

Observed on 2026-08-23: nightly docs identified Rust 1.100.0-nightly dated 2026-08-22 and still marked `portable_simd` experimental.

## `fearless_simd`

- Crate documentation: <https://docs.rs/fearless_simd/latest/fearless_simd/>
- Repository: <https://github.com/linebender/fearless_simd>
- crates.io: <https://crates.io/crates/fearless_simd>

Verified on 2026-09-25: `fearless_simd` 1.0.0, released 2026-09-21, and
`fearless_simd_macros` 0.1.0. Both declare Rust 1.89 as their MSRV.

- Versioned core API: <https://docs.rs/fearless_simd/1.0.0/fearless_simd/>
- Versioned macro guidance: <https://docs.rs/fearless_simd_macros/0.1.0/fearless_simd_macros/>
- Release and migration details: <https://github.com/linebender/fearless_simd/blob/v1.0.0/CHANGELOG.md>

## `wide`

- Crate documentation: <https://docs.rs/wide/latest/wide/>
- Repository: <https://github.com/Lokathor/wide>
- crates.io: <https://crates.io/crates/wide>

Observed on 2026-08-23: crate version `1.6.1`. Read each vector type's docs for arithmetic, reduction, NaN, and conversion semantics.

## Intel x86/x86_64

- Intel Intrinsics Guide: <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html>
- Intel 64 and IA-32 Architectures Software Developer Manuals: <https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html>
- Rust x86_64 intrinsics: <https://doc.rust-lang.org/core/arch/x86_64/index.html>
- Rust x86 intrinsics: <https://doc.rust-lang.org/core/arch/x86/index.html>

Frequently used exact Rust pages:

- `_mm_add_ps`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm_add_ps.html>
- `_mm_add_epi32`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm_add_epi32.html>
- `_mm256_add_ps`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_add_ps.html>
- `_mm256_add_epi32`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_add_epi32.html>
- `_mm_load_ps`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm_load_ps.html>
- `_mm_loadu_ps`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm_loadu_ps.html>
- `_mm256_load_ps`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_load_ps.html>
- `_mm256_loadu_ps`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_loadu_ps.html>
- `_mm_sad_epu8`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm_sad_epu8.html>
- `_mm_adds_epu8`: <https://doc.rust-lang.org/core/arch/x86_64/fn._mm_adds_epu8.html>

Do not use the Intel Intrinsics Guide as a performance promise. It reports instruction metadata and warns that some intrinsics expand to sequences. Validate on the deployment microarchitecture.

## Arm NEON / Advanced SIMD

- Arm C Language Extensions (ACLE): <https://arm-software.github.io/acle/main/acle.html>
- Arm Neon Intrinsics Reference: <https://arm-software.github.io/acle/neon_intrinsics/advsimd.html>
- Rust AArch64 intrinsics: <https://doc.rust-lang.org/core/arch/aarch64/index.html>
- Rust Arm intrinsics: <https://doc.rust-lang.org/core/arch/arm/index.html>
- Arm learning article on Neon programming: <https://community.arm.com/arm-community-blogs/b/architectures-and-processors-blog/posts/coding-for-neon---part-1-load-and-stores>

Frequently used exact Rust pages:

- `vld1q_f32`: <https://doc.rust-lang.org/core/arch/aarch64/fn.vld1q_f32.html>
- `vst1q_f32`: <https://doc.rust-lang.org/core/arch/aarch64/fn.vst1q_f32.html>
- `vld1q_u32`: <https://doc.rust-lang.org/core/arch/aarch64/fn.vld1q_u32.html>
- `vst1q_u32`: <https://doc.rust-lang.org/core/arch/aarch64/fn.vst1q_u32.html>
- `vabdq_u8`: <https://doc.rust-lang.org/core/arch/aarch64/fn.vabdq_u8.html>
- `vpaddlq_u8`: <https://doc.rust-lang.org/core/arch/aarch64/fn.vpaddlq_u8.html>
- `vld3q_u8`: <https://doc.rust-lang.org/core/arch/aarch64/fn.vld3q_u8.html>
- `vbslq_u8`: <https://doc.rust-lang.org/core/arch/aarch64/fn.vbslq_u8.html>

Use ACLE feature macros and Rust target features to distinguish baseline Advanced SIMD from optional extensions such as dot product, FP16, BF16, or SVE.

## Version Review Procedure

Before changing the skill:

1. Check the current stable and nightly Rust documentation.
2. Check crate release pages and MSRV declarations.
3. Re-run the examples on x86_64 and AArch64 where possible.
4. Inspect assembly for generic, SSE2, AVX, AVX2, and NEON builds.
5. Update `metadata.last-verified` and this file's observed versions.
6. Do not silently rewrite semantics to match a newer API.
