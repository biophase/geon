import numpy as np

from geon._native import region_growing


def test_seeded_grower_returns_component_for_seed():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, 120).astype(np.float32)
    y = rng.uniform(-1.0, 1.0, 120).astype(np.float32)
    z = np.zeros(120, dtype=np.float32)
    coords = np.stack([x, y, z], axis=1)
    normals = np.zeros_like(coords)
    normals[:, 2] = 1.0

    grower = region_growing.SeededGrower(
        coords,
        normals,
        normal_mode="use_provided",
        params={
            "epsilon": 0.2,
            "tau": 20,
            "alpha_deg": 35.0,
            "perform_cca": True,
            "enable_seed_gating": False,
        },
    )
    indices, stats = grower.grow(0)
    assert indices.dtype == np.int32
    assert indices.ndim == 1
    assert indices.size >= 20
    assert int(stats["seed_index"]) == 0
    assert bool(stats["accepted"]) is True
    assert 0 in set(indices.tolist())


def test_seeded_grower_rejects_out_of_range_seed():
    coords = np.zeros((10, 3), dtype=np.float32)
    grower = region_growing.SeededGrower(
        coords,
        normal_mode="compute",
        params={"epsilon": 0.1, "tau": 1, "enable_seed_gating": False},
    )
    try:
        grower.grow(10)
    except RuntimeError as exc:
        assert "seed_index out of range" in str(exc)
    else:
        raise AssertionError("Expected out-of-range seed to raise RuntimeError")
