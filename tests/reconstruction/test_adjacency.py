import unittest
import numpy as np
from scipy.spatial import cKDTree
from geon.algorithms.reconstruction.core import Patch,Config,_candidate_pairs
class AdjacencyTests(unittest.TestCase):
    def test_oriented_box_filter_keeps_exact_radius_graph(self):
        rng=np.random.default_rng(5);patches={}
        for i in range(16):
            axis,_=np.linalg.qr(rng.normal(size=(3,3)));points=rng.normal(size=(100,3))*[1,.1,.005];points=points@axis.T+rng.normal(size=3)
            patches[i]=Patch(i,None,points.mean(axis=0),axis[:,2],axis[:,:2],points,None,None,0.)
        cfg=Config(adjacency_radius=.3,coplanar_boundary_tolerance=.3,min_contact_points=5)
        actual={frozenset(p) for p in _candidate_pairs(patches,cfg)};expected=set()
        for i in patches:
            for j in patches:
                if j<=i:continue
                a,b=patches[i],patches[j]
                if np.count_nonzero(cKDTree(b.points).query(a.points)[0]<.3)>=5 and np.count_nonzero(cKDTree(a.points).query(b.points)[0]<.3)>=5:expected.add(frozenset((i,j)))
        self.assertEqual(actual,expected)
if __name__=='__main__':unittest.main()
