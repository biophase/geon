import unittest
import numpy as np
from geon.algorithms.reconstruction.core import Config,reconstruct

class MeshStitchingTests(unittest.TestCase):
    def result(self,gap=.02,enabled=True):
        a,b=np.meshgrid(np.linspace(0,1,26),np.linspace(0,1,26));a=a.ravel();b=b.ravel()
        left=np.column_stack([a,b,a*0]);right=np.column_stack([.3+.4*a,1+gap+b,b*0])
        cfg=Config(alpha_radius=.1,join_coplanar=False,bridge_remaining_gaps=False,stitch_mesh_boundaries=False)
        r=reconstruct(np.vstack([left,right]),np.repeat([1,2],len(a)),config=cfg)
        if enabled:
            from geon.algorithms.reconstruction.stitching import stitch_boundaries
            values=stitch_boundaries(list(r.vertices),list(r.faces),list(r.face_patch_ids),list(r.face_part_ids),list(r.face_is_bridge),r.patches)
            v,f,ids,parts,flags,bridges,_=values
            r.vertices=np.asarray(v);r.faces=np.asarray(f);r.face_patch_ids=np.asarray(ids);r.face_part_ids=np.asarray(parts);r.face_is_bridge=np.asarray(flags);r.bridges.extend(bridges)
        return r
    def test_partial_overlap_inserts_nodes_and_fills_only_gap(self):
        r=self.result()
        self.assertTrue(any(b.get('bridge_type')=='mesh_boundary' for b in r.bridges))
        self.assertEqual(int(r.face_is_bridge.sum()),2)
        self.assertTrue(np.all(r.face_patch_ids[r.face_is_bridge]==1))
        vertices=r.vertices[np.unique(r.faces[r.face_is_bridge])]
        self.assertGreaterEqual(vertices[:,0].min(),.3-1e-8);self.assertLessEqual(vertices[:,0].max(),.7+1e-8)
        faces=r.faces;edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        _,count=np.unique(edges,axis=0,return_counts=True);self.assertTrue(np.all(count<=2))
    def test_coincident_borders_are_noded_without_extra_faces(self):
        r=self.result(gap=0)
        self.assertFalse(r.face_is_bridge.any())
        left=set(r.faces[r.face_patch_ids==1].ravel());right=set(r.faces[r.face_patch_ids==2].ravel())
        self.assertEqual(len(left&right),2)
    def test_overlapping_faces_not_bridged(self):
        r=self.result(gap=-.02)
        self.assertFalse(r.bridges)
    def test_disable(self):
        r=self.result(enabled=False)
        self.assertFalse(r.bridges)

if __name__=='__main__':unittest.main()
