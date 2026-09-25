# Primary-source index

Research context: 2026-09-06. Versioned APIs below are the intended compilation
pins. These links document the recommendations; unversioned branch links are
explicitly not a claim that their current SHA was fetched for this patch. The
preparation environment could not retrieve the target repository or install a
Rust toolchain. Fearless entries were updated against the 1.0.0 release and
compiled registry sources on 2026-09-25. See `VECTORIZE-V2-STATUS.md` for the
current checks and the historical preparation limits.

| ID | Primary source | Used for |
|---|---|---|
| R1 | https://doc.rust-lang.org/std/simd/num/trait.SimdFloat.html | FP reducers, NaN and signed zero |
| R2 | https://doc.rust-lang.org/std/simd/num/trait.SimdUint.html | integer reducers, casts |
| R3 | https://docs.rs/fearless_simd/1.0.0/fearless_simd/trait.SimdBase.html | vector construction, permutations and numeric reducers |
| R4 | https://docs.rs/fearless_simd/1.0.0/fearless_simd/trait.SimdMask.html | mask reductions and bit order |
| R5 | https://github.com/Lokathor/wide/blob/main/src/f32x8_.rs | reducers and compile-time representation |
| R6 | https://docs.rs/wide/1.7.0/wide/struct.f32x8.html | concrete wide API |
| R7 | https://llvm.org/docs/Vectorizers.html | loop/SLP diagnostics, reductions, unrolling |
| R8 | https://en.algorithmica.org/hpc/algorithms/prefix/ | scan and cache blocking |
| R9 | https://github.com/BurntSushi/memchr/blob/master/src/arch/generic/memchr.rs | block byte search |
| R10 | https://github.com/simdjson/simdjson/blob/master/src/generic/stage1/json_string_scanner.h | mask/state pipeline |
| R11 | https://github.com/simdutf/simdutf/blob/master/src/generic/utf8_validation/utf8_lookup4_algorithm.h | lookup-based validation |
| R12 | https://github.com/zlib-ng/zlib-ng/blob/develop/arch/x86/adler32_avx2.c | block checksum algebra |
| R13 | https://github.com/ARM-software/optimized-routines/blob/master/math/aarch64/advsimd/log2f.c | vector log2 construction |
| R14 | https://github.com/shibatch/sleef | production vector math and accuracy |
| R15 | https://github.com/linebender/fearless_simd/blob/main/fearless_simd/examples/srgb.rs | mixed portable/specialized design |
| R16 | https://github.com/fast-pack/streamvbyte | layout-driven integer decoding |
| R17 | https://doc.rust-lang.org/rustc/codegen-options/index.html | remarks and codegen flags |
| R18 | https://doc.rust-lang.org/cargo/reference/config.html | flag precedence and Cargo config |
| R19 | https://doc.rust-lang.org/std/simd/struct.Simd.html | scatter is not scatter-add |
| R20 | https://github.com/linebender/fearless_simd/blob/v1.0.0/CHANGELOG.md | v1 migration, additions and numeric contracts |
| R21 | https://docs.rs/fearless_simd_macros/0.1.0/fearless_simd_macros/ | optional `#[simd]`, token carriers and execution boundaries |

No external source code, tables or polynomial coefficients are copied. Local
examples are original educational implementations of the described algorithms.
When importing future source material, add immutable revision and license data;
never invent provenance or benchmark measurements.

## CI integration sources

- https://github.com/actions/checkout — checked usage selects v6.
- https://github.com/actions/upload-artifact — checked usage selects v7.
- https://github.blog/changelog/2026-01-29-arm64-standard-runners-are-now-available-in-private-repositories/ — standard ARM64 runner labels.

The new workflow uses read-only repository permissions and does not publish or
push code. CI configuration is supplied but was not executed during preparation.
