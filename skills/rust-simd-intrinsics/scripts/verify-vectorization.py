#!/usr/bin/env python3
"""Explicit stable/nightly validation. Missing tools never count as passing tests."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / 'examples/vectorization-v2/Cargo.toml'

def command(tool: str, toolchain: str, *args: str) -> list[str]:
    return [tool] + ([] if toolchain == 'system' else [f'+{toolchain}']) + list(args)

def digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

class Runner:
    def __init__(self, output: Path, strict: bool):
        self.output, self.strict = output, strict
        self.checks: list[dict] = []
        self.output.mkdir(parents=True, exist_ok=True)
    def add(self, name: str, status: str, **details) -> bool:
        entry = {'id': name, 'status': status, **details}
        self.checks.append(entry)
        print(f'{status:11} {name}', flush=True)
        return status == 'passed'
    def unavailable(self, name: str, reason: str) -> bool:
        return self.add(name, 'failed' if self.strict else 'skipped', reason=reason)
    def run(self, name: str, cmd: list[str], cwd: Path = ROOT) -> bool:
        log = self.output / f'{len(self.checks):03d}-{name}.log'
        print('+ ' + ' '.join(cmd), flush=True)
        try:
            with log.open('w', encoding='utf-8') as handle:
                result = subprocess.run(cmd, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False)
            return self.add(name, 'passed' if result.returncode == 0 else 'failed',
                            command=cmd, exit_code=result.returncode, log=str(log))
        except OSError as exc:
            return self.add(name, 'failed', command=cmd, reason=str(exc))

def discover_legacy_features(manifest: Path) -> tuple[list[str], list[str]]:
    data = tomllib.loads(manifest.read_text(encoding='utf-8'))
    features = data.get('features', {})
    nightly = [name for name in features if 'nightly' in name.lower() or 'portable' in name.lower()]
    # Restrict stable options to the existing SIMD libraries rather than assuming
    # that --all-features is safe on stable. Expand nightly dependencies transitively.
    def needs_nightly(name: str, seen: set[str] | None = None) -> bool:
        seen = set() if seen is None else seen
        if name in seen:
            return False
        seen.add(name)
        if name in nightly:
            return True
        return any(needs_nightly(child, seen) for child in features.get(name, []) if child in features)
    stable = [name for name in features if any(x in name.lower() for x in ('fearless', 'wide')) and not needs_nightly(name)]
    return stable, nightly

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='missing required tools or locks fail this scope')
    parser.add_argument('--scope', choices=['stable', 'nightly', 'all'], default='all')
    parser.add_argument('--toolchain', default='stable', help='stable toolchain name, or system for non-rustup Cargo')
    parser.add_argument('--nightly-toolchain', default='nightly')
    parser.add_argument('--bootstrap-lock', action='store_true', help='generate a lock only when absent; never update an existing one')
    parser.add_argument('--legacy', action='store_true', help='also test the original examples/Cargo.toml if present')
    parser.add_argument('--fmt', action='store_true', help='add rustfmt --check (optional, reported separately)')
    parser.add_argument('--output', type=Path, default=ROOT/'.artifacts/vectorization-v2/verification')
    args = parser.parse_args()
    output = args.output.resolve()
    run = Runner(output, args.strict)
    manifests = [('recipes', RECIPE)]
    if args.legacy and (ROOT/'examples/Cargo.toml').is_file():
        manifests.append(('legacy', ROOT/'examples/Cargo.toml'))
    metadata: dict = {
        'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'host': platform.platform(), 'machine': platform.machine(),
        'scope': args.scope, 'strict': args.strict,
        'execution': 'host only; cross-compilation is not an execution result',
        'flags': {k: v for k, v in os.environ.items() if k in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS','RUSTC_WRAPPER') or (k.startswith('CARGO_TARGET_') and k.endswith('_RUSTFLAGS'))},
        'toolchains': {},
    }
    if shutil.which('cargo') is None:
        run.unavailable('cargo', 'Cargo not installed; no Rust compilation or execution took place')
    elif shutil.which('rustc') is None:
        run.unavailable('rustc', 'rustc not installed; no Rust compilation or execution took place')
    else:
        scopes = ['stable', 'nightly'] if args.scope == 'all' else [args.scope]
        for scope in scopes:
            toolchain = args.toolchain if scope == 'stable' else args.nightly_toolchain
            version_cmd = command('cargo', toolchain, '--version')
            version = subprocess.run(version_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            if version.returncode:
                run.unavailable(f'{scope}-toolchain', version.stdout.strip())
                continue
            rustc = subprocess.run(command('rustc', toolchain, '-Vv'), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
            metadata['toolchains'][scope] = {'cargo': version.stdout.strip(), 'rustc': rustc.stdout.strip()}
            if rustc.returncode:
                run.unavailable(f'{scope}-rustc', rustc.stdout.strip())
                continue
            for label, manifest in manifests:
                if not manifest.is_file():
                    run.add(f'{scope}-{label}-manifest', 'failed', reason=f'missing {manifest}')
                    continue
                lock = manifest.with_name('Cargo.lock')
                if not lock.is_file() and args.bootstrap_lock:
                    if not run.run(f'{scope}-{label}-bootstrap-lock', command('cargo', toolchain, 'generate-lockfile', '--manifest-path', str(manifest))):
                        continue
                if not lock.is_file():
                    run.unavailable(f'{scope}-{label}-lock', 'Cargo.lock missing; resolve with --bootstrap-lock, review and commit it')
                    continue
                if label == 'recipes':
                    combinations = ['', 'fearless', 'wide', 'fearless,wide'] if scope == 'stable' else ['portable', 'portable,fearless,wide']
                    defaults = ['--no-default-features']
                else:
                    stable, nightly = discover_legacy_features(manifest)
                    combinations = [''] + stable if scope == 'stable' else [','.join(stable + nightly)]
                    # Preserve the legacy workspace's default contract.
                    defaults = []
                for features in combinations:
                    slug = (features or 'default').replace(',', '-')
                    for profile in ['debug', 'release']:
                        cmd = command('cargo', toolchain, 'test', '--locked', '--all-targets', '--manifest-path', str(manifest), *defaults)
                        if features:
                            cmd += ['--features', features]
                        if profile == 'release':
                            cmd += ['--release']
                        cmd += ['--', '--nocapture']
                        run.run(f'{scope}-{label}-{slug}-{profile}', cmd)
                if args.fmt and scope == 'stable':
                    run.run(f'{scope}-{label}-format', command('cargo', toolchain, 'fmt', '--manifest-path', str(manifest), '--', '--check'))
    metadata['manifests'] = [{'path': str(p.relative_to(ROOT)), 'manifest_sha256': digest(p), 'lock_sha256': digest(p.with_name('Cargo.lock'))} for _, p in manifests]
    counts = {s: sum(c['status'] == s for c in run.checks) for s in ['passed','failed','skipped','unsupported']}
    complete = bool(run.checks) and counts['failed'] == 0 and counts['skipped'] == 0 and counts['unsupported'] == 0
    report = {**metadata, 'status': 'passed' if complete else ('failed' if counts['failed'] else 'incomplete'),
              'complete_for_requested_scope': complete, 'counts': counts, 'checks': run.checks,
              'performance_claim': None}
    report_path = output/'verification.json'
    report_path.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'counts': counts, 'report': str(report_path)}), flush=True)
    return 1 if counts['failed'] else 0

if __name__ == '__main__':
    raise SystemExit(main())
