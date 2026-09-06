#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLES="$ROOT/examples/Cargo.toml"

required=(
  "SKILL.md"
  "LICENSE"
  "README.md"
  "references/platforms-and-instructions.md"
  "references/auto-vectorization.md"
  "references/libraries.md"
  "references/algorithm-cookbook.md"
  "references/correctness-benchmarking.md"
  "references/source-index.md"
  "examples/Cargo.toml"
  "examples/src/lib.rs"
  "examples/src/autovec.rs"
  "examples/src/std_arch.rs"
  "examples/src/wide_impl.rs"
  "examples/src/fearless_impl.rs"
  "examples/src/portable_simd.rs"
)

for file in "${required[@]}"; do
  if [[ ! -f "$ROOT/$file" ]]; then
    echo "missing required file: $file" >&2
    exit 1
  fi
done

if ! grep -q '^name: rust-simd-intrinsics$' "$ROOT/SKILL.md"; then
  echo "SKILL.md frontmatter name is missing or incorrect" >&2
  exit 1
fi

if [[ $(wc -l < "$ROOT/SKILL.md") -gt 500 ]]; then
  echo "SKILL.md exceeds the 500-line context budget" >&2
  exit 1
fi

if grep -RInE '\b(TODO|FIXME|TBD)\b|PLACEHOLDER' \
  "$ROOT/SKILL.md" "$ROOT/references" "$ROOT/examples/src"; then
  echo "unresolved placeholder found" >&2
  exit 1
fi

echo "skill structure: OK"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found: Rust compilation checks skipped"
  exit 0
fi

if command -v rustfmt >/dev/null 2>&1; then
  cargo fmt --manifest-path "$EXAMPLES" --all -- --check
fi

cargo test --manifest-path "$EXAMPLES" --release
cargo test --manifest-path "$EXAMPLES" --release --features wide-example
cargo test --manifest-path "$EXAMPLES" --release --features fearless-example

if command -v rustup >/dev/null 2>&1 && rustup toolchain list | grep -q '^nightly'; then
  cargo +nightly test \
    --manifest-path "$EXAMPLES" \
    --release \
    --features portable-simd-example
else
  echo "nightly toolchain not found: portable std::simd check skipped"
fi

echo "all available checks passed"
