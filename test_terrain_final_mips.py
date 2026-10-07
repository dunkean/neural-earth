import unittest
import numpy as np
from terrain_final_mips import plan_mip,read_mip


class FinalMipTests(unittest.TestCase):
    def test_signed_global_parent_mean_and_negative_coordinates(self):
        plan=plan_mip(2,-1,-2)
        def child(tx,ty):
            xs=tx*256+np.arange(-24,280)
            ys=ty*256+np.arange(-24,280)
            return (xs[None,:]*2+ys[:,None]*3).astype(np.float32)
        actual=read_mip(plan,child)
        xs=plan.x0+np.arange(plan.size)*plan.step+(plan.step-1)/2
        ys=plan.y0+np.arange(plan.size)*plan.step+(plan.step-1)/2
        np.testing.assert_array_equal(actual,xs[None,:]*2+ys[:,None]*3)
        self.assertLess(actual.max(),0)

    def test_missing_child_never_publishes_partial_mip_and_memory_is_bounded(self):
        plan=plan_mip(1,0,0)
        calls=[]
        def child(tx,ty):
            calls.append((tx,ty))
            return None if (tx,ty)==plan.children[1] else np.zeros((304,304),np.float32)
        self.assertIsNone(read_mip(plan,child))
        self.assertEqual(len(calls),2)
        self.assertIsNone(plan_mip(12,0,0))


if __name__=='__main__':unittest.main()
