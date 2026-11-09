import unittest
import numpy as np
from geon.algorithms.reconstruction.core import Config,reconstruct,_edge_points

class CoplanarTests(unittest.TestCase):
    def data(self, offset=0., tilt=0., gap=.03, same_side=False):
        a,b=np.meshgrid(np.linspace(0,1,26),np.linspace(0,1,26));a=a.ravel();b=b.ravel()
        x=a if same_side else a+1+gap
        xyz=np.vstack([np.column_stack([a,b,a*0]),np.column_stack([x,b,offset+tilt*(x-1)])])
        return xyz,np.repeat([1,2],len(a))

    def check_mesh(self,result):
        edges=[e for e in result.shared_edges if e.get('kind')=='coplanar']
        self.assertTrue(edges,result.diagnostics)
        self.assertEqual(set(result.face_patch_ids),{1,2})
        for edge in edges:
            curve=_edge_points(edge)
            indices=[]
            for point in curve:
                index=np.argmin(np.linalg.norm(result.vertices-point,axis=1))
                self.assertLess(np.linalg.norm(result.vertices[index]-point),1e-7)
                indices.append(index)
            for label in edge['patch_ids']:
                faces=result.faces[result.face_patch_ids==label]
                for a,b in zip(indices,indices[1:]):
                    self.assertTrue(np.any(np.isin(faces,[a,b]).sum(axis=1)==2))
        return edges

    def test_adjacent_faces_share_border_and_remain_separate(self):
        xyz,ids=self.data()
        r=reconstruct(xyz,ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07))
        self.check_mesh(r)
        self.assertTrue(all(p.polygon.is_valid for p in r.patches.values()))

    def test_slight_offset_and_tilt_weld_exactly(self):
        xyz,ids=self.data(offset=.004,tilt=.005)
        r=reconstruct(xyz,ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07))
        self.check_mesh(r)

    def test_parallel_offset_rejected(self):
        xyz,ids=self.data(offset=.03)
        r=reconstruct(xyz,ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07))
        self.assertFalse(r.shared_edges)

    def test_duplicate_same_side_borders_rejected(self):
        xyz,ids=self.data(same_side=True)
        r=reconstruct(xyz,ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07))
        self.assertFalse(r.shared_edges)

    def test_disable_and_part_restriction(self):
        xyz,ids=self.data()
        r=reconstruct(xyz,ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07,join_coplanar=False))
        self.assertFalse(r.shared_edges)
        r=reconstruct(xyz,ids,part_ids=ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07))
        self.assertFalse(r.shared_edges)

    def test_curved_shared_border_is_simplified_once(self):
        y,t=np.meshgrid(np.linspace(0,1,51),np.linspace(0,1,26))
        y,t=y.ravel(),t.ravel()
        seam=1+.15*np.sin(2*np.pi*y)
        left=np.column_stack([t*seam,y,y*0])
        right=np.column_stack([seam+.02+t*(2-seam-.02),y,y*0])
        xyz=np.vstack([left,right]);ids=np.repeat([1,2],len(y))
        r=reconstruct(xyz,ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07,simplify_tolerance=.01))
        edges=self.check_mesh(r)
        self.assertTrue(any(len(_edge_points(e))>2 for e in edges))

    def test_only_overlapping_border_is_matched(self):
        a,b=np.meshgrid(np.linspace(0,1,26),np.linspace(0,1,26));a=a.ravel();b=b.ravel()
        xyz=np.vstack([np.column_stack([a,b,a*0]),np.column_stack([a+1.03,b+.4,a*0])])
        r=reconstruct(xyz,np.repeat([1,2],len(a)),config=Config(coplanar_join_mode="snap", alpha_radius=.07))
        edges=self.check_mesh(r)
        for e in edges:
            self.assertGreaterEqual(e['points'][:,1].min(),.4-1e-8)
            self.assertLessEqual(e['points'][:,1].max(),1+1e-8)

    def test_coplanar_border_tolerance_independent_of_intersection_tolerance(self):
        xyz,ids=self.data()
        r=reconstruct(xyz,ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07,edge_tolerance=.005,coplanar_boundary_tolerance=.08))
        self.check_mesh(r)

    def test_closed_shared_ring_replaces_whole_boundary(self):
        from types import SimpleNamespace
        from shapely.geometry import box
        from geon.algorithms.reconstruction.core import _batch_boundary,_has_curve
        patch=SimpleNamespace(footprint=box(0,0,1,1),project=lambda x:np.asarray(x)[:,:2])
        curve=np.array([[-.01,-.01,0],[1.01,-.01,0],[1.01,1.01,0],[-.01,1.01,0],[-.01,-.01,0]])
        edge={'points':curve,'endpoints':curve[[0,-1]],'kind':'coplanar'}
        geometry=_batch_boundary(patch,[edge],Config(coplanar_join_mode="snap", ))
        self.assertIsNotNone(geometry)
        self.assertTrue(geometry.is_valid)
        self.assertTrue(_has_curve(geometry,curve[:,:2]))

    def test_coplanar_search_uses_its_own_boundary_tolerance(self):
        xyz,ids=self.data(gap=.06)
        r=reconstruct(xyz,ids,config=Config(coplanar_join_mode="snap", alpha_radius=.07,adjacency_radius=.03,coplanar_boundary_tolerance=.1))
        self.check_mesh(r)

if __name__=='__main__':unittest.main()
