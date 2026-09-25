# Rust SIMD Intrinsics — Agent Skill

[![skills.sh](https://skills.sh/b/Mnwa/rust-simd-intrinsics)](https://skills.sh/Mnwa/rust-simd-intrinsics)

A reusable Agent Skills package for designing, implementing, reviewing, and benchmarking optimized Rust SIMD code across:

- x86/x86_64 SSE (SSE1), SSE2, AVX, and AVX2;
- Arm/AArch64 NEON (Advanced SIMD);
- LLVM auto-vectorization;
- `std::arch` intrinsics;
- `fearless_simd`;
- `wide`;
- nightly portable `std::simd`.

## Install

Install with the [skills CLI](https://skills.sh/docs/cli):

```bash
npx skills add Mnwa/rust-simd-intrinsics
```

Alternatively, copy the `rust-simd-intrinsics` directory into the skills directory used by your LLM agent. Keep the directory name unchanged because it matches the `name` in `SKILL.md`.

## Layout

```text
rust-simd-intrinsics/
├── SKILL.md
├── README.md
├── references/
│   ├── platforms-and-instructions.md
│   ├── auto-vectorization.md
│   ├── libraries.md
│   ├── algorithm-cookbook.md
│   ├── correctness-benchmarking.md
│   └── source-index.md
├── examples/
│   ├── Cargo.toml
│   ├── README.md
│   └── src/
└── scripts/
    ├── verify.sh
    └── emit-asm.sh
```

`SKILL.md` is the concise operational specification. References are loaded progressively for architecture, crate, algorithm, and verification details. The example crate contains safe scalar, explicit portable, and architecture-specific templates.

## Design Principles

- Scalar oracle and profiling before explicit SIMD.
- Exact semantics for overflow, floating point, masks, and tails.
- Runtime feature dispatch outside hot loops.
- Scalar fallback for heterogeneous deployment.
- Small unsafe boundaries with local safety proofs.
- Assembly inspection and real benchmark evidence before performance claims.

## Toolchains

The core and `wide`/`fearless_simd` examples target stable Rust; the pinned `fearless_simd` 1.0.0 example requires Rust 1.89 or newer. The `std::simd` module is feature-gated and requires nightly Rust with `portable_simd`.

## Validate

Run the following from this skill directory (`skills/rust-simd-intrinsics/`
inside the repository).

```bash
./scripts/verify.sh
```

The script validates the package structure and, when Cargo is installed, checks/tests the stable examples and optional library examples. Nightly portable SIMD checks are attempted only when a nightly toolchain is present.

## Version Snapshot

Documentation and library observations were last reviewed on 2026-08-23. See `references/source-index.md` for primary sources and the exact snapshot notes.

## License

[MIT](LICENSE), copyright 2026 Mnwa.
