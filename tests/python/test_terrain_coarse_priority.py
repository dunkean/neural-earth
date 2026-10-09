"""Real coarse window planning: camera focus, complete land then sea, replay."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import tempfile
import unittest
from test_terrain_windows import _world, _manifest
from terrain_coarse import CoarsePreparation, COARSE_METRES as CELL


class CoarsePriorityTests(unittest.TestCase):
    def test_zoom_reorders_land_before_sea_and_still_completes_ocean(self):
        with tempfile.TemporaryDirectory() as directory:
            world=_world(cache_bytes=100000)
            bounds=(0,0,16*CELL,8*CELL)
            preparation=CoarsePreparation(directory,_manifest(947),bounds=bounds,
                async_persistence=False).install(world)
            try:
                mask={'bounds':bounds,'width':16,'height':8,'rows':['1111000000000000']*8}
                # Even a sea-centred camera cannot discard/deprioritize all
                # remaining land behind ocean in the full-world idle queue.
                preparation.prioritize(mask,(12*CELL,2*CELL,16*CELL,6*CELL))
                geometry={entry[0]:entry for entry in preparation._priority_geometry}
                self.assertFalse(geometry[preparation._priority_order[0]][1])
                classes=[geometry[index][1] for index in preparation._priority_order]
                self.assertEqual(classes,sorted(classes))
                self.assertIn(True,classes)
                self.assertEqual(set(preparation._priority_order),set(preparation._indices))
                first=preparation._priority_order[0]
                preparation.step(world)
                self.assertIn(first,preparation._persisted_indices)
                preparation.prioritize(mask,(0,0,2*CELL,2*CELL))
                second=next(index for index in preparation._priority_order if index not in preparation._persisted_indices)
                self.assertNotEqual(second,first)
                previous=len(world.calls['coarse'])
                preparation.step(world)
                self.assertEqual(world.calls['coarse'][previous:], [second])
                result=preparation.step(world,budget_windows=len(preparation._indices))
                self.assertEqual(result['complete_windows'],result['total_windows'])
                self.assertTrue(any(geometry[index][1] for index in world.calls['coarse']))
                self.assertEqual(len(world.calls['coarse']),len(set(world.calls['coarse'])))
            finally:
                preparation.close()


if __name__=='__main__':unittest.main()
