import numpy as np

from geon.algorithms.reconstruction.core import Config, reconstruct


def test_clean_first_three_cropped_planes_share_one_corner():
    u, v = np.meshgrid(np.linspace(.03, 1, 26), np.linspace(.03, 1, 26))
    u, v = u.ravel(), v.ravel()
    xyz = np.vstack([np.column_stack([u, v, u*0]),
                     np.column_stack([u, u*0, v]),
                     np.column_stack([u*0, u, v])])
    result = reconstruct(xyz, np.repeat([1, 2, 3], len(u)),
                         config=Config(clean_first=True, alpha_radius=.07,
                                       corner_tolerance=.1))
    assert len(result.shared_edges) == 3
    corner = np.flatnonzero(np.linalg.norm(result.vertices, axis=1) < 1e-7)
    assert len(corner) == 1
    for label in (1, 2, 3):
        assert corner[0] in result.faces[result.face_patch_ids == label]
    edges = np.sort(np.concatenate([result.faces[:, [0, 1]],
                                    result.faces[:, [1, 2]],
                                    result.faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert counts.max() == 2


def test_clean_first_preserves_observed_hole_in_output_mesh(monkeypatch):
    import geon.algorithms.reconstruction.core as core
    def late_simplification(*args, **kwargs):
        raise AssertionError('Clean boundaries must not be simplified again after joining')
    monkeypatch.setattr(core, '_simplify', late_simplification)
    u, v = np.meshgrid(np.linspace(0, 1, 26), np.linspace(0, 1, 26))
    u, v = u.ravel(), v.ravel()
    keep = ~((u > .3) & (u < .7) & (v > .3) & (v < .7))
    xyz = np.column_stack([u[keep], v[keep], u[keep]*0])
    result = reconstruct(xyz, np.ones(len(xyz), dtype=int),
                         config=Config(clean_first=True, alpha_radius=.045))
    polygon = result.patches[1].polygon
    assert len(polygon.interiors) == 1
    triangles = result.vertices[result.faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1).sum()/2
    np.testing.assert_allclose(area, polygon.area)
