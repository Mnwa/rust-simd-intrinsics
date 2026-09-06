# Apply and validate the vectorization v2 upgrade

Run these commands from a real, locally available checkout of
`Mnwa/rust-simd-intrinsics`. Python 3.11 or newer and Git are needed for integration.
Rust/Cargo, rustup-managed stable/nightly toolchains and network access for the
initial dependency resolution are needed for the Rust verification steps.

## 1. Apply the additive patch

```bash
git status --short
git apply --check /path/to/rust-simd-intrinsics-vectorization-v2.patch
git apply /path/to/rust-simd-intrinsics-vectorization-v2.patch
```

The new paths must not already contain conflicting files. If `--check` fails,
inspect the conflicts instead of forcing the patch. No upstream base commit was
available during preparation; see [the validation status](VECTORIZE-V2-STATUS.md).

## 2. Integrate with the existing skill

The following command preserves the existing skill text and adds the new routing.
It also updates the existing verification/assembly entry points. It refuses to
replace dirty, staged or untracked original files that it would change. Commit
or otherwise preserve your local changes first; do not delete them to bypass this
check. The new files from step 1 may remain untracked.

```bash
python3 scripts/integrate-vectorization-v2.py --write
git diff --check
git diff -- SKILL.md references scripts/verify.sh scripts/emit-asm.sh .gitignore
```

The tool is idempotent and does not create commits, push, install dependencies or
modify remote state. Existing script contents are replaced by delegates to the
new verifier/codegen tools; original Rust examples and dependency versions are
not rewritten. Review any local/custom script behavior before adopting them.

To inspect and apply the integration **as a second ordinary Git patch** instead:

```bash
python3 scripts/integrate-vectorization-v2.py > /tmp/simd-v2-integration.patch
git apply --check /tmp/simd-v2-integration.patch
git apply /tmp/simd-v2-integration.patch
```

Use either the `--write` route or the second-patch route, not both. After staging,
`git diff --cached` includes the new files as well as the original-file changes.
An optional single combined patch against your actual HEAD can then be generated
with `git diff --cached --binary --full-index` in a dedicated clean branch/index.
Do not use a shared index containing unrelated staged work.

## 3. Run available non-Rust checks

```bash
python3 -m unittest discover -s scripts/tests -v
python3 scripts/test-vectorization-models.py
python3 scripts/score-vectorization-evals.py evals/vectorization-v2/example-unmeasured.json
```

The last input is an explicitly unmeasured schema/example fixture, not evidence
that an agent experiment ran or improved performance.

## 4. Resolve and verify Rust

```bash
python3 scripts/verify-vectorization.py --strict --scope all --bootstrap-lock --legacy
```

The verifier generates only missing lockfiles. Review and commit actual resolved
lockfiles and record exact compiler versions before relying on reproducibility.
It checks debug/release and relevant feature combinations. Missing tools or locks
fail strict mode; ordinary local mode reports them as skipped, never passed.

The delivered Rust has not been compiled in the authoring environment. Treat any
API/compiler issue as a real failure to fix, not as a reason to disable a test or
relax semantic requirements. Formatting can be checked after applying rustfmt:

```bash
cargo +stable fmt --manifest-path examples/vectorization-v2/Cargo.toml
python3 scripts/verify-vectorization.py --strict --scope all --legacy --fmt
```

## 5. Inspect codegen and measure on deployment hardware

```bash
python3 scripts/emit-vectorization-asm.py generic --features fearless,wide
python3 scripts/emit-vectorization-asm.py native --features fearless,wide
cargo +stable run --release --locked \
  --manifest-path examples/vectorization-v2/Cargo.toml \
  --features fearless,wide --bin recipe-bench -- 65536 7 20
```

Use nightly with `--features portable,fearless,wide` for the portable recipes.
Repeat across small/large sizes, distributions and target CPUs. Keep all samples,
validate correctness first, include caller-visible setup/dispatch costs, and do
not present a Python model or the supplied CI definition as executed Rust tests.
