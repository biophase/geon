import unittest
from types import SimpleNamespace
import numpy as np
from geon.algorithms.reconstruction.core import _close_junction_loops

class JunctionTests(unittest.TestCase):
    def data(self):
        # Tetrahedron missing one tiny triangular side, a closed boundary loop.
        v=[np.array(x) for x in [[0,0,0],[.04,0,0],[0,.04,0],[0,0,.04]]]
        return v,[[0,1,3],[1,2,3],[2,0,3]],[1,2,3],[-1,-1,-1],[True,False,False],{i:SimpleNamespace(id=i,part_id=None,source_point_count=i*10) for i in (1,2,3)}
    def test_tiny_multipatch_bridge_junction_closed(self):
        args=self.data();n=_close_junction_loops(*args,.15)
        self.assertEqual(n,1);self.assertEqual(len(args[1]),4);self.assertEqual(args[2][-1],1)
    def test_no_cap_without_bridge_or_for_single_patch_holes(self):
        args=self.data();args[4][:]=[False]*3
        self.assertEqual(_close_junction_loops(*args,.15),0)
        args=self.data();args[2][:]=[1]*3
        self.assertEqual(_close_junction_loops(*args,.15),0)
    def test_large_open_loop_preserved(self):
        args=self.data()
        self.assertEqual(_close_junction_loops(*args,.01),0)

if __name__=='__main__':unittest.main()
