#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/examples/Cargo.toml"
MODE="${1:-generic}"
TARGET="${TARGET:-}"
FEATURES="${FEATURES:-}"

case "$MODE" in
  generic)
    RUSTFLAGS_VALUE=""
    ;;
  native)
    RUSTFLAGS_VALUE="-C target-cpu=native"
    ;;
  sse2)
    RUSTFLAGS_VALUE="-C target-feature=+sse2"
    ;;
  avx)
    RUSTFLAGS_VALUE="-C target-feature=+avx"
    ;;
  avx2)
    RUSTFLAGS_VALUE="-C target-feature=+avx2"
    ;;
  neon)
    RUSTFLAGS_VALUE="-C target-feature=+neon"
    ;;
  *)
    echo "usage: $0 {generic|native|sse2|avx|avx2|neon}" >&2
    echo "optional environment: TARGET=<triple> FEATURES=<cargo-feature-list>" >&2
    exit 2
    ;;
esac

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required" >&2
  exit 1
fi

cargo_args=(rustc --manifest-path "$MANIFEST" --release --lib)
if [[ -n "$TARGET" ]]; then
  cargo_args+=(--target "$TARGET")
fi
if [[ -n "$FEATURES" ]]; then
  cargo_args+=(--features "$FEATURES")
fi
cargo_args+=(-- --emit=asm)

echo "emitting assembly in mode: $MODE${TARGET:+ for $TARGET}"
if [[ -n "$RUSTFLAGS_VALUE" ]]; then
  RUSTFLAGS="$RUSTFLAGS_VALUE" cargo "${cargo_args[@]}"
else
  cargo "${cargo_args[@]}"
fi

if [[ -n "$TARGET" ]]; then
  TARGET_DIR="$ROOT/examples/target/$TARGET/release/deps"
else
  TARGET_DIR="$ROOT/examples/target/release/deps"
fi

find "$TARGET_DIR" -maxdepth 1 -type f -name '*.s' -print | sort

echo
echo "Inspect complete loops, dispatch guards, tails, spills, and call boundaries."
