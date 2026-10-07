"""Check benchmark evidence and restoration without importing its CUDA app."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


def harness_definitions(optimize=0):
    path=Path(__file__).with_name('benchmark_runtime_lod.py')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    names={'Measurements','compare','require_exact_errors','validate_fidelity',
           'reference_evidence','imported_source_evidence','write_receipt','run_with_receipt'}
    selected=ast.Module(body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names],type_ignores=[])
    namespace=dict(np=np,hashlib=hashlib,json=json,Path=Path,defaultdict=defaultdict)
    exec(compile(selected,str(path),'exec',optimize=optimize),namespace)
    return namespace


class RuntimeBenchmarkCpuTests(unittest.TestCase):
    def test_exact_gate_rejects_signed_zero_and_tolerated_changes(self):
        definitions=harness_definitions()
        compare,require=definitions['compare'],definitions['require_exact_errors']
        expected=np.array([0.,1.],dtype=np.float32)
        identity=compare(expected.copy(),expected)
        require(identity,[identity]*5)
        signed=compare(np.array([-0.,1.],dtype=np.float32),expected)
        self.assertTrue(signed['exact'])
        self.assertFalse(signed['byte_exact'])
        with self.assertRaises(ValueError):
            require(signed,[identity]*5)
        tolerated=compare(np.array([0.,1.00001],dtype=np.float32),expected)
        self.assertLess(tolerated['max_abs'],1.)
        with self.assertRaises(ValueError):
            require(identity,[identity]*4+[tolerated])

    def test_stage_restoration_preserves_exact_saved_functions(self):
        definitions=harness_definitions()
        original=lambda *args: 'original'
        shared=SimpleNamespace(args=[],_f=original,uuid='shared')
        first=SimpleNamespace(args=[shared],_f=lambda *args: 'first',uuid='first')
        root=SimpleNamespace(args=[first,shared],_f=lambda *args: 'root',uuid='root')
        previous=[node._f for node in (shared,first,root)]
        measurement=definitions['Measurements']()
        measurement.install_stages(SimpleNamespace(residual=root))
        self.assertEqual(len(measurement._stage_wrappers),3)
        measurement.restore_stages()
        for node,saved in zip((shared,first,root),previous):
            self.assertIs(node._f,saved)
        measurement.restore_stages()

    def test_reference_provenance_identifies_npz_and_receipt(self):
        evidence=harness_definitions()['reference_evidence']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'reference.npz'
            np.savez(path,elevation=np.zeros((2,2),dtype=np.float32))
            receipt=path.with_suffix('.json')
            receipt.write_text(json.dumps({'label':'frozen-before'}),encoding='utf-8')
            result=evidence(path)
            self.assertEqual(result['path'],str(path.resolve()))
            self.assertEqual(result['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(result['label'],'frozen-before')
            self.assertEqual(result['receipt_sha256'],hashlib.sha256(receipt.read_bytes()).hexdigest())

    def test_source_provenance_hashes_actual_imported_paths(self):
        definitions=harness_definitions()
        with tempfile.TemporaryDirectory() as directory:
            module_names=('terrain_inference','terrain_cuda_graphs','terrain_nn_constants',
                          'terrain_window_scheduler','server','terrain_climate','terrain_app',
                          'terrain_conditioning','terrain_world','terrain_manifest','world_pipeline',
                          'mp_layers','harness')
            modules={}
            for name in module_names:
                path=Path(directory)/(name+'.py')
                path.write_text('# '+name,encoding='utf-8')
                modules[name]=SimpleNamespace(__file__=str(path))
            definitions.update(modules)
            definitions['__name__']='actual_harness'
            definitions['sys']=SimpleNamespace(modules={'actual_harness':modules['harness']})
            result=definitions['imported_source_evidence']()
            for module in modules.values():
                path=Path(module.__file__)
                self.assertEqual(result['source_paths'][path.name],str(path.resolve()))
                self.assertEqual(result['source_sha256'][path.name],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_receipt_records_running_complete_and_failed_sample_under_optimization(self):
        # Extract the actual gate with Python -O semantics; assertions in the
        # harness would disappear here. Keep the failing metrics in the receipt.
        definitions=harness_definitions(optimize=2)
        runner=definitions['run_with_receipt']
        compare,validate=definitions['compare'],definitions['validate_fidelity']
        identity=compare(np.array([0.],dtype=np.float32),np.array([0.],dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run.json'
            report=dict(samples=[])
            def completed():
                self.assertEqual(json.loads(path.read_text())['status'],'running')
                report['samples'].append(dict(key='pass',status='complete'))
            runner(path,report,completed)
            self.assertEqual(json.loads(path.read_text())['status'],'complete')

            def failed():
                item=dict(key='failure',status='running',seconds=1.2,
                          elevation_error_m=compare(np.array([2.],dtype=np.float32),np.array([0.],dtype=np.float32)),
                          climate_errors=[identity]*5)
                report['samples'].append(item)
                validate(item,True)
            with self.assertRaisesRegex(ValueError,'numerical limits'):
                runner(path,report,failed)
            saved=json.loads(path.read_text())
            self.assertEqual(saved['status'],'failed')
            self.assertEqual(saved['failing_sample_key'],'failure')
            self.assertEqual(saved['samples'][-1]['status'],'failed')
            self.assertEqual(saved['samples'][-1]['elevation_error_m']['max_abs'],2.)
            self.assertFalse(saved['samples'][-1]['fidelity_passed'])
            self.assertEqual(saved['samples'][-1]['seconds'],1.2)

            # The exact gate remains active even when the tolerance gate passes.
            signed=compare(np.array([-0.],dtype=np.float32),np.array([0.],dtype=np.float32))
            with self.assertRaisesRegex(ValueError,'byte-identical'):
                validate(dict(elevation_error_m=signed,climate_errors=[identity]*5),True)


if __name__=='__main__':
    unittest.main()
