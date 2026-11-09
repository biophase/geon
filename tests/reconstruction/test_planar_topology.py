"""Run beside the module: python -m unittest -v test_planar_topology.py"""
import unittest
import numpy as np
from geon.algorithms.reconstruction.core import Config, reconstruct


class SharedTopologyTests(unittest.TestCase):
    def faces(self, count=3):
        a, b = np.meshgrid(np.linspace(.03, 1, 26), np.linspace(.03, 1, 26))
        a, b = a.ravel(), b.ravel()
        points = [np.column_stack([a,b,a*0]), np.column_stack([a,a*0,b]), np.column_stack([a*0,a,b])]
        return np.vstack(points[:count]), np.repeat(np.arange(1,count+1),len(a))

    def test_three_cropped_faces_share_corner_and_mesh_edges(self):
        xyz, ids = self.faces()
        result = reconstruct(xyz, ids, config=Config(alpha_radius=.07, corner_tolerance=.1))
        self.assertEqual(len(result.shared_edges), 3)
        corner = np.flatnonzero(np.linalg.norm(result.vertices, axis=1) < 1e-7)
        self.assertEqual(len(corner), 1)
        for i in (1,2,3):
            self.assertIn(corner[0], result.faces[result.face_patch_ids == i])
        for edge in result.shared_edges:
            ends = [np.argmin(np.linalg.norm(result.vertices-p,axis=1)) for p in edge['endpoints']]
            for label in edge['patch_ids']:
                triangles = result.faces[result.face_patch_ids == label]
                self.assertTrue(np.any(np.isin(triangles, ends).sum(axis=1) == 2))
        self.assertTrue(all(p.polygon.is_valid for p in result.patches.values()))

    def test_two_faces_do_not_invent_corner(self):
        xyz, ids = self.faces(2)
        result = reconstruct(xyz, ids, config=Config(alpha_radius=.07))
        self.assertEqual(len(result.shared_edges), 1)
        self.assertGreater(np.linalg.norm(result.shared_edges[0]['endpoints'],axis=1).min(), .02)

    def test_corner_movement_is_bounded(self):
        xyz, ids = self.faces()
        result = reconstruct(xyz, ids, config=Config(alpha_radius=.07, corner_tolerance=.01))
        self.assertFalse(np.any(np.linalg.norm(result.vertices,axis=1) < 1e-7))

    def test_different_parts_remain_separate(self):
        xyz, ids = self.faces()
        result = reconstruct(xyz, ids, part_ids=ids, config=Config(alpha_radius=.07))
        self.assertFalse(result.shared_edges)

    def test_holes_survive(self):
        a,b=np.meshgrid(np.linspace(0,1,26),np.linspace(0,1,26))
        a,b=a.ravel(),b.ravel()
        keep=~((a>.3)&(a<.7)&(b>.3)&(b<.7))
        xyz=np.column_stack([a[keep],b[keep],a[keep]*0])
        result=reconstruct(xyz,np.ones(len(xyz),dtype=int),config=Config(alpha_radius=.045))
        self.assertEqual(len(result.patches[1].polygon.interiors),1)
        triangles=result.vertices[result.faces]
        area=np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=1).sum()/2
        self.assertAlmostEqual(area,result.patches[1].polygon.area)

class InsertedBoundaryTests(unittest.TestCase):
    def test_midsegment_endpoints_extend_boundary(self):
        from types import SimpleNamespace
        from shapely.geometry import box
        from geon.algorithms.reconstruction.core import _batch_boundary, _has_edge
        patch = SimpleNamespace(footprint=box(0, 0, 1, 1), project=lambda x: np.asarray(x)[:, :2])
        edge = {'endpoints': np.array([[.3, -.03, 0], [.7, -.03, 0]])}
        geometry = _batch_boundary(patch, [edge], Config())
        self.assertIsNotNone(geometry)
        self.assertTrue(geometry.is_valid)
        self.assertTrue(_has_edge(geometry, edge['endpoints'][:, :2]))
        self.assertLess(geometry.bounds[1], 0)  # Extension outside observed footprint.

    def test_two_segments_can_share_original_boundary_edge(self):
        from types import SimpleNamespace
        from shapely.geometry import box
        from geon.algorithms.reconstruction.core import _batch_boundary, _has_edge
        patch = SimpleNamespace(footprint=box(0, 0, 1, 1), project=lambda x: np.asarray(x)[:, :2])
        edges = [{'endpoints': np.array([[a, -.03, 0], [b, -.03, 0]])}
                 for a,b in ((.1,.4),(.6,.9))]
        geometry = _batch_boundary(patch, edges, Config())
        self.assertIsNotNone(geometry)
        self.assertTrue(all(_has_edge(geometry,e['endpoints'][:, :2]) for e in edges))

    def test_extension_beyond_tolerance_rejected(self):
        from types import SimpleNamespace
        from shapely.geometry import box
        from geon.algorithms.reconstruction.core import _batch_boundary
        patch = SimpleNamespace(footprint=box(0, 0, 1, 1), project=lambda x: np.asarray(x)[:, :2])
        edge = {'endpoints': np.array([[.3, -.2, 0], [.7, -.2, 0]])}
        self.assertIsNone(_batch_boundary(patch, [edge], Config(edge_tolerance=.08)))

if __name__=='__main__':
    unittest.main()
