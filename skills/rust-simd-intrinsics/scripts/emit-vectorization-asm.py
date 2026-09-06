#!/usr/bin/env python3
"""Emit isolated generic/native codegen, with explicit effective rustflags."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def tool(name: str, chain: str) -> list[str]:
    return [name] + ([] if chain == 'system' else [f'+{chain}'])

def clean_flags(environment: dict[str, str], flags: list[str]) -> dict[str, str]:
    env = environment.copy()
    for key in list(env):
        if key in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'CARGO_BUILD_RUSTFLAGS', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER') or (key.startswith('CARGO_TARGET_') and key.endswith('_RUSTFLAGS')):
            del env[key]
    # Highest-priority Cargo rustflags source, also overriding .cargo config flags.
    env['CARGO_ENCODED_RUSTFLAGS'] = '\x1f'.join(flags)
    return env

def generic_cpu(target: str) -> str:
    return 'x86-64' if target.startswith('x86_64-') else 'generic'

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=['generic','native'], default='generic')
    parser.add_argument('--features', default='')
    parser.add_argument('--toolchain', default='stable')
    parser.add_argument('--target')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if shutil.which('cargo') is None or shutil.which('rustc') is None:
        parser.error('cargo and rustc are required; no assembly was emitted')
    manifest = ROOT/'examples/vectorization-v2/Cargo.toml'
    lock = manifest.with_name('Cargo.lock')
    if not lock.is_file():
        parser.error('generate and review examples/vectorization-v2/Cargo.lock before --locked codegen')
    result = subprocess.run(tool('rustc',args.toolchain)+['-Vv'], text=True, capture_output=True, check=False)
    if result.returncode:
        parser.error(result.stderr.strip() or result.stdout.strip())
    host = next((line.split(': ',1)[1] for line in result.stdout.splitlines() if line.startswith('host: ')), None)
    if host is None:
        parser.error('rustc did not report a host triple')
    target = args.target or host
    if args.mode == 'native' and target != host:
        parser.error('native mode is only meaningful for the host target; use generic for cross compilation')
    cpu = 'native' if args.mode == 'native' else generic_cpu(target)
    flags = ['-C', f'target-cpu={cpu}', '-C', 'debuginfo=1']
    env = clean_flags(dict(os.environ), flags)
    output = (args.output or ROOT/'.artifacts/vectorization-v2/codegen'/args.mode).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configs = [Path.home()/'.cargo/config', Path.home()/'.cargo/config.toml']
    if os.environ.get('CARGO_HOME'):
        configs += [Path(os.environ['CARGO_HOME'])/'config', Path(os.environ['CARGO_HOME'])/'config.toml']
    metadata = {'mode':args.mode,'target':target,'cpu':cpu,'features':args.features,'rustc':result.stdout,
                'effective_rustflags':flags,'inherited_flag_variables':{k:v for k,v in os.environ.items() if 'RUSTFLAGS' in k},
                'cargo_home_configs': [{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in configs if p.is_file()],
                'lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),'commands':[], 'status':'failed'}
    exit_code = 1
    with tempfile.TemporaryDirectory(prefix='simd-v2-codegen-') as directory:
        stage=Path(directory)/'workspace'
        shutil.copytree(manifest.parent,stage,ignore=shutil.ignore_patterns('target','.git','.artifacts'))
        env['CARGO_TARGET_DIR']=str(Path(directory)/'target')
        common=tool('cargo',args.toolchain)+['--config','build.rustc-wrapper=""','--config','build.rustc-workspace-wrapper=""',
            'rustc','--locked','--release','--target',target,'--manifest-path',str(stage/'Cargo.toml'),'--no-default-features']
        if args.features:
            common += ['--features',args.features]
        exit_code=0
        for label,target_args in [('library',['--lib']),('driver',['--bin','recipe-bench'])]:
            cmd=common+target_args+['--','-C','remark=all','--emit=asm,llvm-ir']
            metadata['commands'].append(cmd)
            print('+ '+' '.join(cmd),flush=True)
            with (output/f'{label}.log').open('w',encoding='utf-8') as log:
                build=subprocess.run(cmd,cwd=stage,env=env,stdout=log,stderr=subprocess.STDOUT,check=False)
            if build.returncode:
                exit_code=build.returncode
                break
        artifacts=[]
        for suffix in ('*.s','*.ll'):
            for path in sorted(Path(env['CARGO_TARGET_DIR']).rglob(suffix)):
                rel=path.relative_to(Path(env['CARGO_TARGET_DIR']))
                destination=output/'files'/rel
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,destination)
                artifacts.append(str(destination.relative_to(output)))
        if exit_code==0 and not artifacts:
            exit_code=1
            metadata['error']='compiler succeeded but emitted no .s/.ll files'
        metadata['artifacts']=artifacts
    metadata['status']='passed' if exit_code==0 else 'failed'
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f"{metadata['status']}: {output}")
    return exit_code

if __name__=='__main__':
    raise SystemExit(main())
