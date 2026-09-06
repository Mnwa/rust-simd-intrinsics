#!/usr/bin/env python3
"""Generate/apply a repository-specific integration diff without guessing its base."""
from __future__ import annotations
import argparse
import difflib
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
START='<!-- vectorization-v2:start -->'
END='<!-- vectorization-v2:end -->'
ROUTING='''## Vectorization v2: required algorithm-selection supplement

Before editing SIMD code, read [the v2 workflow](references/vectorization-v2.md).
It adds executable reductions, scans, mask pipelines, lane-axis selection and
real-world case studies. Load only the relevant detailed reference from its table.
For missing reducers, compare a built-in operation, a composed helper, a final
lane-array epilogue and a small ISA specialization; do not invent APIs.

The supplement supersedes older advice about numeric reduction API names,
`wide` caller-only target-feature dispatch, inherited generic-build flags and
silent toolchain skips. All original memory-safety and semantic requirements
remain in force. Reassociation requires an explicit numeric contract.

Runnable implementations and differential tests are in the independent workspace
`examples/vectorization-v2`. Run `python3 scripts/verify-vectorization.py --strict --scope all --bootstrap-lock --legacy`
in a connected Rust environment, review the
new lockfile and commit it. Later runs must reuse the lock without updating it.
See `references/vectorization-validation.md` for scope, backend and benchmark rules.
Never report a skipped check, an uncompiled candidate or a cross-build as a passing
runtime test. Preserve scalar-written autovectorized performance baselines.
'''
SUPPLEMENTS={
    'references/libraries.md': '''## Version-specific capabilities and dispatch correction

Read [the API matrix](api-capabilities.md) before choosing or naming methods.
`std::simd` uses `reduce_product`, `wide::f32x8` uses `reduce_mul`, and the
Fearless 0.7 examples supply numeric reducers while reusing built-in mask reducers.
A `#[target_feature]` annotation on a caller does not re-evaluate `wide`'s compiled
`cfg(target_feature)` branches. Prefer a measured portable kernel plus a small
specialized helper when a single operation is missing.
''',
    'references/auto-vectorization.md': '''## Diagnose before replacing the algorithm

Read [vectorization patterns](vectorization-patterns.md) and consider more than one
lane axis. Loop dependencies can require scan/block algebra instead of surrendering
to scalar code. Some strict FP reductions can be vectorized without reassociation;
independent-output lanes are another option. `rustc -C remark=all` emits documented
optimization remarks. Use `scripts/emit-vectorization-asm.py generic` and `native`
for isolated flags, explicit CPU baselines, metadata, assembly and LLVM IR.
''',
    'references/algorithm-cookbook.md': '''## Executable transformation recipes

The [v2 routing table](vectorization-v2.md) links reductions, prefix scans,
argmin/argmax, masks, compaction, histograms, checksums, vector math and lane-axis
examples. Each recipe has scalar oracles and tests in `examples/vectorization-v2`.
Read [reductions](reductions.md), [patterns](vectorization-patterns.md), and
[case studies](case-studies.md) before writing a new missing primitive.
''',
    'references/correctness-benchmarking.md': '''## Required validation additions

Follow [the v2 validation contract](vectorization-validation.md): lengths through
257 plus large cases, slice offsets, backend-specific execution, integer overflow,
FP contracts, and strict missing-tool handling. Keep correctness pass rate separate
from speedups among successful candidates. End-to-end measurements must include
packing/setup/dispatch costs that the actual caller pays.
''',
}

def inject(text:str,body:str,frontmatter:bool=False)->str:
    block=f'{START}\n{body.rstrip()}\n{END}'
    if START in text:
        if text.count(START)!=1 or text.count(END)!=1:
            raise ValueError('ambiguous or unterminated integration markers')
        a=text.index(START); b=text.index(END,a)+len(END)
        return text[:a]+block+text[b:]
    if END in text:
        raise ValueError('orphaned integration end marker')
    position=0
    if frontmatter:
        lines=text.splitlines(keepends=True)
        if lines and lines[0].lstrip('\ufeff').strip()=='---':
            end=next((i for i in range(1,len(lines)) if lines[i].strip()=='---'),None)
            if end is None:
                raise ValueError('unterminated SKILL.md YAML frontmatter')
            position=sum(map(len,lines[:end+1]))
    return text[:position]+('\n' if position else '')+block+'\n\n'+text[position:]

def plans(root:Path)->dict[Path,str]:
    for name in ['SKILL.md','references/vectorization-v2.md','scripts/verify-vectorization.py','scripts/emit-vectorization-asm.py']:
        if not (root/name).is_file():
            raise ValueError(f'missing required file: {name}; apply the additive patch in the skill repository first')
    targets={root/'SKILL.md':inject((root/'SKILL.md').read_text(encoding='utf-8'),ROUTING,True)}
    for relative,body in SUPPLEMENTS.items():
        path=root/relative
        if path.is_file():
            targets[path]=inject(path.read_text(encoding='utf-8'),body)
    targets[root/'scripts/verify.sh']='''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/verify-vectorization.py" --legacy "$@"
'''
    targets[root/'scripts/emit-asm.sh']='''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/emit-vectorization-asm.py" "$@"
'''
    ignore=root/'.gitignore'
    before=ignore.read_text(encoding='utf-8') if ignore.is_file() else ''
    if '/.artifacts/vectorization-v2/' not in before.splitlines():
        targets[ignore]=before+('' if not before or before.endswith('\n') else '\n')+'/.artifacts/vectorization-v2/\n'
    return {p:value for p,value in targets.items() if not p.exists() or p.read_text(encoding='utf-8')!=value}

def check_path(root:Path,path:Path)->None:
    if not path.is_relative_to(root):
        raise ValueError('target escapes repository')
    for part in [path,*path.parents]:
        if part==root:
            break
        if part.is_symlink():
            raise ValueError(f'refusing symlink integration target: {part}')

def patch_text(root:Path,changes:dict[Path,str])->str:
    result=[]
    for path,value in changes.items():
        relative=path.relative_to(root).as_posix()
        old=path.read_text(encoding='utf-8') if path.exists() else ''
        result.append(f'diff --git a/{relative} b/{relative}\n')
        if not path.exists():
            result.append('new file mode '+('100755' if path.suffix=='.sh' else '100644')+'\n')
        diff=list(difflib.unified_diff(old.splitlines(keepends=True),value.splitlines(keepends=True),
             fromfile=f'a/{relative}' if path.exists() else '/dev/null',tofile=f'b/{relative}'))
        for line in diff:
            if line.endswith('\n'):
                result.append(line)
            else:
                result.append(line+'\n\\ No newline at end of file\n')
    return ''.join(result)

def atomic_write(path:Path,text:str,mode:int)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    name=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as handle:
            name=handle.name
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.chmod(name,mode)
        os.replace(name,path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)

def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='apply integration to clean tracked files; otherwise print a normal git diff')
    args=parser.parse_args()
    try:
        top=subprocess.run(['git','rev-parse','--show-toplevel'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
        if Path(top).resolve()!=ROOT:
            raise ValueError('run this script from the skill checkout, not a nested or unrelated repository')
        # Reject symlinks before reading any existing integration target.
        for relative in ['SKILL.md','.gitignore','scripts/verify.sh','scripts/emit-asm.sh',*SUPPLEMENTS]:
            check_path(ROOT,ROOT/relative)
        changes=plans(ROOT)
        if not args.write:
            sys.stdout.write(patch_text(ROOT,changes))
            return 0
        if not changes:
            print('Integration already current; no files changed.')
            return 0
        paths=[str(p.relative_to(ROOT)) for p in changes]
        dirty=subprocess.run(['git','status','--porcelain','--untracked-files=all','--',*paths],cwd=ROOT,text=True,capture_output=True,check=True).stdout
        if dirty:
            raise ValueError('refusing to overwrite dirty/untracked integration targets; commit or stash those files first:\n'+dirty)
        originals={p:(p.read_bytes(),stat.S_IMODE(p.stat().st_mode)) if p.exists() else None for p in changes}
        written=[]
        try:
            for path,value in changes.items():
                mode=originals[path][1] if originals[path] else (0o755 if path.suffix=='.sh' else 0o644)
                atomic_write(path,value,mode); written.append(path)
        except Exception:
            for path in reversed(written):
                original=originals[path]
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(original[0]); path.chmod(original[1])
            raise
        print('Integrated '+str(len(changes))+' files. Review with git diff; no commit was created.')
        return 0
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print(f'error: {exc}',file=sys.stderr)
        return 1

if __name__=='__main__':
    raise SystemExit(main())
