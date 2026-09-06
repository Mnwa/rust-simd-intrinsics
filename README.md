# Rust SIMD Intrinsics — Agent Skill

[![skills.sh](https://skills.sh/b/Mnwa/rust-simd-intrinsics)](https://skills.sh/Mnwa/rust-simd-intrinsics)

Guidance and executable examples for designing, reviewing, and benchmarking Rust
SIMD code across x86/x86_64 and Arm NEON, including auto-vectorization,
`std::arch`, `fearless_simd`, `wide`, and nightly `std::simd`.

## Install

```bash
npx skills add Mnwa/rust-simd-intrinsics
```

For manual installation, copy [skills/rust-simd-intrinsics](skills/rust-simd-intrinsics)
into your agent's skills directory. Keep the folder name unchanged.

See the [skill documentation](skills/rust-simd-intrinsics/README.md) for the
layout, toolchains, and design principles.

## Validate

Run from the repository root:

```bash
gh skill publish --dry-run
./skills/rust-simd-intrinsics/scripts/verify.sh
```

The skill lives in a named subdirectory so GitHub CLI can validate its name.
Run `gh skill publish --dry-run` from the repository root; passing the skill
directory itself triggers the [root-level name validation issue](https://github.com/cli/cli/issues/13284)
in GitHub CLI 2.97.0.

## License

[MIT](LICENSE), copyright 2026 Mnwa. A copy is bundled with the skill.
