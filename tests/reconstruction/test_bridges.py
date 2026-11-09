import tempfile
import unittest
from pathlib import Path
import numpy as np
from plyfile import PlyData
from geon.algorithms.reconstruction.core import Config,reconstruct

class BridgeTests(unittest.TestCase):
    def data(self,gap=.03,offset=0,tilt=0):
        a,b=np.meshgrid(np.linspace(0,1,26),np.linspace(0,1,26));a=a.ravel();b=b.ravel()
        c,d=np.meshgrid(np.linspace(0,1,18),np.linspace(0,1,18));c=c.ravel()+1+gap;d=d.ravel()
        xyz=np.vstack([np.column_stack([a,b,a*0]),np.column_stack([c,d,offset+tilt*(c-1)])])
        return xyz,np.r_[np.ones(len(a),dtype=int),np.full(len(c),2,dtype=int)]

    def check_bridge(self,r):
        self.assertTrue(r.bridges,r.diagnostics)
        self.assertTrue(r.face_is_bridge.any())
        self.assertTrue(np.all(r.face_patch_ids[r.face_is_bridge]==2))
        for patch in r.patches.values():
            points=r.vertices[np.unique(r.faces[(r.face_patch_ids==patch.id)&~r.face_is_bridge])]
            self.assertLess(np.abs((points-patch.center)@patch.normal).max(),1e-8)
        for bridge in r.bridges:
            self.assertEqual(bridge['owner_id'],2)
            for label,curve in bridge['side_points'].items():
                source=r.faces[(r.face_patch_ids==label)&~r.face_is_bridge]
                strip=r.faces[r.face_is_bridge]
                indices=[]
                for point in curve:
                    k=np.argmin(np.linalg.norm(r.vertices-point,axis=1))
                    self.assertLess(np.linalg.norm(r.vertices[k]-point),1e-7)
                    indices.append(k)
                for a,b in zip(indices,indices[1:]):
                    self.assertTrue(np.any(np.isin(source,[a,b]).sum(axis=1)==2))
                    self.assertTrue(np.any(np.isin(strip,[a,b]).sum(axis=1)==2))

    def test_coplanar_gap_filled_and_owned_by_smaller_patch(self):
        xyz,ids=self.data()
        r=reconstruct(xyz,ids,config=Config(alpha_radius=.07))
        self.check_bridge(r)
        triangles=r.vertices[r.faces[r.face_is_bridge]]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2
        self.assertGreater(area,0)

    def test_tilted_offset_faces_stay_on_planes(self):
        xyz,ids=self.data(offset=.004,tilt=.005)
        r=reconstruct(xyz,ids,config=Config(alpha_radius=.07))
        self.check_bridge(r)

    def test_overlap_is_not_bridged(self):
        xyz,ids=self.data(gap=-.03)
        r=reconstruct(xyz,ids,config=Config(alpha_radius=.07))
        self.assertFalse(r.bridges)
        self.assertFalse(r.face_is_bridge.any())

    def test_ply_retains_bridge_ownership_and_marker(self):
        xyz,ids=self.data()
        r=reconstruct(xyz,ids,config=Config(alpha_radius=.07))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bridge.ply';r.export_ply(path)
            faces=PlyData.read(path)['face'].data
            self.assertTrue(np.array_equal(faces['is_bridge'],r.face_is_bridge))
            self.assertTrue(np.all(faces['patch_id'][faces['is_bridge'].astype(bool)]==2))

    def test_vertical_transition_with_no_projected_gap(self):
        xyz,ids=self.data(gap=0,offset=.004)
        r=reconstruct(xyz,ids,config=Config(alpha_radius=.07))
        self.check_bridge(r)

    def test_subsampling_keeps_original_size_ownership(self):
        xyz,ids=self.data()
        r=reconstruct(xyz,ids,config=Config(alpha_radius=.09,subsample_distance=.05))
        self.check_bridge(r)
        self.assertEqual(r.patches[1].source_point_count,676)
        self.assertEqual(r.patches[2].source_point_count,324)

    def test_crossing_strip_rejected(self):
        from types import SimpleNamespace
        from shapely.geometry import box
        from geon.algorithms.reconstruction.core import _bridge_is_valid
        patch=lambda i:SimpleNamespace(id=i,project=lambda p:np.asarray(p)[:,:2],lift=lambda p:np.column_stack([p,np.zeros(len(p))]))
        edge={'patch_ids':(1,2),'side_points':{1:np.array([[0,0,0],[1,0,0]]),2:np.array([[1,1,0],[0,1,0]])}}
        self.assertFalse(_bridge_is_valid(edge,{1:patch(1),2:patch(2)},{1:box(-1,-1,2,0),2:box(-1,1,2,2)},Config()))

if __name__=='__main__':unittest.main()
