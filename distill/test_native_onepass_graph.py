"""One-pass capture wrapper preserves CPU windows and has no CPU graph cache."""
import unittest

import torch

from distill.native_onepass import noise_windows
from distill.native_onepass_graph import OnePassGraph
from distill.native_onepass_runtime import OnePassRuntime
from distill.test_native_onepass import tiny_saved


class OnePassGraphTests(unittest.TestCase):
    def test_cpu_eager_outputs_are_identical(self):
        torch.manual_seed(14)
        runtime=OnePassRuntime(tiny_saved())
        graph=OnePassGraph(runtime)
        noises=noise_windows(613457,[(0,0),(0,32)])
        condition=torch.randn(2,58)
        with torch.no_grad():
            self.assertTrue(torch.equal(graph(noises,condition),runtime(noises,condition)))
        self.assertIsNone(graph._cache)


if __name__=='__main__':unittest.main()
