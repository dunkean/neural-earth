import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from distill.common import REVISION
from distill.native_site_manifest import capture_site,freeze,load_sites


class NativeSitesTest(unittest.TestCase):
    def make(self,root,split='final',seed=101):
        metadata=dict(split=split,seed=seed,teacher_revision=REVISION,native_size=128,
            x=-96,y=32,profile='natural')
        p=root/'capture.npz';np.savez(p,metadata=json.dumps(metadata|dict(stage_counts={'first':49,'second':25})))
        (root/'manifest.json').write_text(json.dumps(dict(revision=REVISION,
            scenes=[dict(file=p.name,metadata=metadata,case_kind='wet-high-mountain')])))
        return p

    def test_frozen_coordinates_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);p=self.make(root);output=root/'sites.json'
            freeze([p],output);sites=load_sites(output)
            self.assertEqual((sites[0]['x'],sites[0]['y']),(-7680,23040))
            manifest=json.loads(output.read_text());manifest['sites'][0]['x']+=240
            output.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load_sites(output)

    def test_excludes_train_and_reserved_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for split,seed in [('train',10300),('val',101),('final',10300)]:
                p=self.make(root,split,seed)
                with self.assertRaises(ValueError):capture_site(p)


if __name__=='__main__':unittest.main()
