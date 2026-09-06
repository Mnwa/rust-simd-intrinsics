"""Tests for patch integration, flag isolation, verification and eval accounting.
These do NOT compile Rust or validate CPU intrinsics.
"""
from __future__ import annotations
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]

def load(name:str,filename:str):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

integration=load('integration','integrate-vectorization-v2.py')
codegen=load('codegen','emit-vectorization-asm.py')
scorer=load('scorer','score-vectorization-evals.py')

class IntegrationTests(unittest.TestCase):
    def test_frontmatter_is_preserved_and_integration_is_idempotent(self):
        original='---\nname: fixture\ndescription: Synthetic test, not upstream\n---\n# Original\nKeep me.\n'
        changed=integration.inject(original,'## New\nInstructions\n',True)
        self.assertTrue(changed.startswith(original.split('# Original')[0]))
        self.assertIn('# Original\nKeep me.',changed)
        self.assertEqual(changed,integration.inject(changed,'## New\nInstructions\n',True))
    def test_malformed_markers_and_yaml_fail(self):
        for text in ['---\nname: bad\n',integration.START+'\nno end',integration.END]:
            with self.assertRaises(ValueError):
                integration.inject(text,'body',True)
    def test_symlink_targets_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'SKILL.md').symlink_to(root/'outside')
            with self.assertRaises(ValueError):
                integration.check_path(root,root/'SKILL.md')
    def test_generated_git_diff_applies_to_real_file_contents(self):
        # A synthetic repository tests the mechanics, not compatibility with upstream.
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            subprocess.run(['git','init','-q'],cwd=root,check=True)
            old={
                'SKILL.md':'---\nname: synthetic-fixture\n---\n# Keep original content',
                'references/libraries.md':'# Libraries\nExisting text.\n',
                'scripts/verify.sh':'#!/bin/sh\nexit 0\n',
            }
            for name,text in old.items():
                path=root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(text)
            subprocess.run(['git','add','.'],cwd=root,check=True)
            for name in ['references/vectorization-v2.md','scripts/verify-vectorization.py','scripts/emit-vectorization-asm.py']:
                path=root/name; path.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,path)
            changes=integration.plans(root)
            patch=integration.patch_text(root,changes)
            for args in [['git','apply','--check','-'],['git','apply','-']]:
                subprocess.run(args,cwd=root,input=patch,text=True,check=True,capture_output=True)
            for path,text in changes.items():
                self.assertEqual(path.read_text(),text)
            self.assertEqual(integration.plans(root),{})
            for name in ['scripts/verify.sh','scripts/emit-asm.sh']:
                subprocess.run(['bash','-n',str(root/name)],check=True)

    def make_cli_fixture(self, root, repo=None):
        subprocess.run(['git','init','-q'],cwd=repo or root,check=True)
        (root/'SKILL.md').write_text('---\nname: synthetic-fixture\n---\n# Original\nRetain this text.\n')
        (root/'scripts').mkdir()
        (root/'scripts/verify.sh').write_text('#!/bin/sh\nexit 0\n')
        subprocess.run(['git','add','.'],cwd=root,check=True)
        subprocess.run(['git','-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
                        '-c','commit.gpgsign=false','commit','-qm','synthetic integration fixture'],cwd=root,check=True)
        for name in ['scripts/integrate-vectorization-v2.py','scripts/verify-vectorization.py',
                     'scripts/emit-vectorization-asm.py','references/vectorization-v2.md']:
            target=root/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        return root/'scripts/integrate-vectorization-v2.py'
    def test_cli_write_then_repeat_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); script=self.make_cli_fixture(root)
            first=subprocess.run([sys.executable,str(script),'--write'],cwd=root,text=True,capture_output=True)
            self.assertEqual(first.returncode,0,first.stderr)
            before=(root/'SKILL.md').read_bytes()
            second=subprocess.run([sys.executable,str(script),'--write'],cwd=root,text=True,capture_output=True)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertIn('already current',second.stdout)
            self.assertEqual(before,(root/'SKILL.md').read_bytes())
            self.assertIn('Retain this text.',before.decode())
    def test_cli_refuses_dirty_original_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); script=self.make_cli_fixture(root)
            (root/'SKILL.md').write_text((root/'SKILL.md').read_text()+'Local uncommitted edit.\n')
            before=(root/'SKILL.md').read_bytes()
            result=subprocess.run([sys.executable,str(script),'--write'],cwd=root,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('dirty',result.stderr)
            self.assertEqual(before,(root/'SKILL.md').read_bytes())

    def test_cli_integrates_nested_skill_and_preserves_dirty_files(self):
        with tempfile.TemporaryDirectory() as directory:
            repo=Path(directory)
            root=repo/'skills/synthetic-fixture'
            root.mkdir(parents=True)
            script=self.make_cli_fixture(root,repo)
            preview=subprocess.run([sys.executable,str(script)],cwd=repo,text=True,capture_output=True)
            self.assertEqual(preview.returncode,0,preview.stderr)
            subprocess.run(['git','apply','--check','-'],cwd=root,input=preview.stdout,text=True,check=True,capture_output=True)
            first=subprocess.run([sys.executable,str(script),'--write'],cwd=repo,text=True,capture_output=True)
            self.assertEqual(first.returncode,0,first.stderr)
            self.assertIn(integration.START,(root/'SKILL.md').read_text())
            (root/'SKILL.md').write_text('---\nname: synthetic-fixture\n---\n# Local edit\n')
            before=(root/'SKILL.md').read_bytes()
            result=subprocess.run([sys.executable,str(script),'--write'],cwd=repo,text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('dirty',result.stderr)
            self.assertEqual(before,(root/'SKILL.md').read_bytes())

class FlagTests(unittest.TestCase):
    def test_generic_flags_remove_both_rustflag_sources(self):
        original={'RUSTFLAGS':'-Ctarget-cpu=native','CARGO_ENCODED_RUSTFLAGS':'-C\x1ftarget-feature=+avx2',
                  'CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS':'-Ctarget-cpu=native',
                  'CARGO_BUILD_RUSTFLAGS':'bad','RUSTC_WRAPPER':'custom','OTHER':'keep'}
        snapshot=original.copy()
        result=codegen.clean_flags(original,['-C','target-cpu=x86-64'])
        self.assertEqual(original,snapshot)
        self.assertEqual(result,{'OTHER':'keep','CARGO_ENCODED_RUSTFLAGS':'-C\x1ftarget-cpu=x86-64'})
    def test_explicit_generic_cpu_names(self):
        self.assertEqual(codegen.generic_cpu('x86_64-unknown-linux-gnu'),'x86-64')
        self.assertEqual(codegen.generic_cpu('aarch64-apple-darwin'),'generic')

class VerificationTests(unittest.TestCase):
    def invoke_without_tools(self,strict:bool):
        with tempfile.TemporaryDirectory() as directory:
            args=[sys.executable,str(ROOT/'scripts/verify-vectorization.py'),'--output',directory]
            if strict:
                args.append('--strict')
            env=os.environ.copy(); env['PATH']=directory
            result=subprocess.run(args,env=env,capture_output=True,text=True)
            report=json.loads((Path(directory)/'verification.json').read_text())
            return result,report
    def test_missing_cargo_is_failure_in_strict_mode(self):
        result,report=self.invoke_without_tools(True)
        self.assertNotEqual(result.returncode,0)
        self.assertEqual(report['status'],'failed')
        self.assertFalse(report['complete_for_requested_scope'])
        self.assertEqual(report['counts']['passed'],0)
    def test_missing_cargo_is_visible_skip_locally(self):
        result,report=self.invoke_without_tools(False)
        self.assertEqual(result.returncode,0)
        self.assertEqual(report['status'],'incomplete')
        self.assertGreater(report['counts']['skipped'],0)
        self.assertEqual(report['counts']['passed'],0)

class EvaluationTests(unittest.TestCase):
    def base(self):
        return json.loads((ROOT/'evals/vectorization-v2/example-unmeasured.json').read_text())[0]
    def test_unmeasured_example_does_not_become_a_speedup(self):
        report=scorer.summarize([self.base()])
        group=report['variants']['updated-skill']
        self.assertEqual(group['correct_and_safe_pass_rate'],0)
        self.assertIsNone(group['geomean_speedup_correct_only'])
    def test_failed_and_unsafe_attempts_remain_in_denominator(self):
        good=self.base()
        good.update(run_id='good',compile_status='passed',correctness_status='passed',baseline_ns=10,candidate_ns=5)
        bad=copy.deepcopy(good); bad.update(run_id='bad',seed=1,safety_violations=1)
        report=scorer.summarize([good,bad])['variants']['updated-skill']
        self.assertEqual(report['correct_and_safe_pass_rate'],0.5)
        self.assertAlmostEqual(report['geomean_speedup_correct_only'],2.0)
        self.assertEqual(report['measured_correct_attempts'],1)
    def test_invalid_and_nonfinite_measurements_rejected(self):
        for value in [0,-1,float('nan'),float('inf'),True]:
            row=self.base(); row.update(baseline_ns=value,candidate_ns=1)
            with self.assertRaises(ValueError):
                scorer.validate([row])
    def test_paired_configurations_are_not_silently_mixed(self):
        rows=[]
        for variant in scorer.VARIANTS:
            row=self.base(); row.update(run_id=variant,variant=variant)
            rows.append(row)
        report=scorer.summarize(rows)
        self.assertEqual(report['paired_attempt_keys'],1)
        self.assertFalse(report['warnings'])
        rows[0]['seed']=42
        self.assertTrue(scorer.summarize(rows)['warnings'])

if __name__=='__main__':
    unittest.main()
