"""Importer integration tests; optional readers are needed for file fixtures."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData


spec = importlib.util.spec_from_file_location(
    "import_laser_scans", Path(__file__).parents[1] / "scripts/import_laser_scans.py"
)
importer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(importer)


@pytest.mark.parametrize("normalize", [True, False])
def test_mixed_formats_roundtrip(tmp_path, normalize):
    pye57 = pytest.importorskip("pye57")
    laspy = pytest.importorskip("laspy")
    pytest.importorskip("lazrs")
    origin = np.array([500000., 5400000., 100.])
    xyz = np.array([[0., 0., 0.], [0.002, 0., 0.], [0.03, 0., 0.]])
    raw = {f"cartesian{axis}": xyz[:, i] for i, axis in enumerate("XYZ")}
    raw.update({key: np.full(3, val, dtype=np.uint8) for key, val in
                zip(("colorRed", "colorGreen", "colorBlue"), (255, 128, 0))})
    # Third return is invalid; pose rotates x into y and translates to world.
    raw["cartesianInvalidState"] = np.array([0, 0, 1], dtype=np.int8)
    raw["intensity"] = np.array([12.5, 42.25, 99.75], dtype=np.float32)
    with pye57.E57(str(tmp_path / "a.E57"), mode="w") as writer:
        writer.write_scan_raw(raw, translation=origin,
                              rotation=np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)]))
        writer.write_scan_raw(
            {f"cartesian{axis}": xyz[:, i] for i, axis in enumerate("XYZ")},
            translation=origin + [1., 0., 0.],
        )
    worlds = [origin + [[0., 0., 0.], [0., .002, 0.]], origin + xyz + [1., 0., 0.]]
    for index, suffix in enumerate(("las", "laz")):
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = origin
        header.scales = np.full(3, .0001)
        las = laspy.LasData(header)
        world = origin + xyz + [2. + index, 0., 0.]
        las.x, las.y, las.z = world.T
        las.intensity = np.array([0, 1234, 65535], dtype=np.uint16)
        las.red, las.green, las.blue = (np.full(3, v, dtype=np.uint16) for v in (65535, 32896, 0))
        las.write(tmp_path / f"b{index}.{suffix}")
        worlds.append(world)
    sources = list(tmp_path.iterdir())
    hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in sources}
    results = importer.import_folder(tmp_path, normalize_intensity=normalize)
    assert [output.name for output, _ in results] == ["a.h5", "b0.h5", "b1.h5"]
    output, sidecar = results[0]
    doc = Document.load_hdf5(output)
    cloud = next(iter(doc.scene_items.values()))
    meta = json.loads(sidecar.read_text())
    mean = np.concatenate(worlds[:2]).mean(axis=0)
    np.testing.assert_allclose(meta["world_mean_before_subsampling"], mean, rtol=0, atol=1e-8)
    assert cloud.points.dtype == np.float32
    assert cloud["scan_id"].dtype == np.uint32
    assert cloud["color"].dtype == np.uint8
    assert doc.name == "a"
    assert set(cloud["scan_id"].ravel()) == {0, 1}
    assert meta["point_count_before_subsampling"] == 5
    assert len(cloud.points) == 3
    assert cloud["intensity"].dtype == (np.float64 if normalize else np.float32)
    np.testing.assert_array_equal(cloud["intensity"].ravel(), [0 if normalize else 12.5, 0, 0])
    assert meta["intensity_normalization"]["enabled"] == normalize
    assert meta["intensity_normalization"]["applied"] == normalize
    if normalize:
        assert meta["intensity_normalization"]["offset"] == 12.5
    np.testing.assert_array_equal(cloud["intensity_valid"].ravel(), [1, 0, 0])
    assert cloud.get_fields(names="intensity")[0].field_type == FieldType.INTENSITY
    assert [scan["has_intensity"] for scan in meta["scans"]] == [True, False]
    restored = cloud.points.astype(np.float64) + mean
    for scan_id, world in enumerate(worlds[:2]):
        rows = cloud["scan_id"].ravel() == scan_id
        np.testing.assert_allclose(restored[rows], world[[0] if scan_id == 0 else [0, 2]], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(cloud["rgb_valid"][rows], 0 if scan_id == 1 else 1)
        expected = [0, 0, 0] if scan_id == 1 else [255, 128, 0]
        np.testing.assert_array_equal(cloud["color"][rows], np.tile(expected, (rows.sum(), 1)))
    distances = np.linalg.norm(cloud.points[:, None] - cloud.points[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    assert distances.min() >= .01
    assert json.loads(doc.meta["laser_scan_import"]) == meta
    for index, (output, sidecar) in enumerate(results[1:]):
        doc = Document.load_hdf5(output)
        cloud = next(iter(doc.scene_items.values()))
        meta = json.loads(sidecar.read_text())
        assert doc.name == f"b{index}"
        assert set(cloud.field_names) == {"color", "intensity"}
        assert cloud["intensity"].dtype == (np.float64 if normalize else np.uint16)
        np.testing.assert_array_equal(cloud["intensity"].ravel(), [0, 1 if normalize else 65535])
        assert cloud.get_fields(names="intensity")[0].field_type == FieldType.INTENSITY
        assert cloud.points.dtype == np.float32
        assert cloud["color"].dtype == np.uint8
        assert "scan_id" not in meta["scans"][0]
        mean = worlds[index + 2].mean(axis=0)
        np.testing.assert_allclose(meta["world_mean_before_subsampling"], mean, rtol=0, atol=1e-8)
        np.testing.assert_allclose(cloud.points.astype(np.float64) + mean,
                                   worlds[index + 2][[0, 2]], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(cloud["color"], [[255, 128, 0], [255, 128, 0]])
        assert meta["point_count_before_subsampling"] == 3
        assert meta["point_count_after_subsampling"] == 2
    assert all(hashlib.sha256(p.read_bytes()).digest() == digest for p, digest in hashes.items())
    with pytest.raises(FileExistsError):
        importer.import_folder(tmp_path)


def test_discovery_centering_and_invalid_points(tmp_path, monkeypatch):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "scan.LAS").touch()
    with pytest.raises(ValueError, match="No E57"):
        importer.import_folder(tmp_path)
    points = np.array([[1e6, 2., 3.], [1e6 + .02, 4., 5.], [np.nan, 0., 0.]])
    monkeypatch.setattr(importer, "read_scans", lambda path: iter([
        (0, points, None, np.array([10, 20, 30], dtype=np.uint16))
    ]))
    output_dir = tmp_path / "outputs"
    [(output, sidecar)] = importer.import_folder(tmp_path, output_dir, recursive=True)
    assert output == output_dir / "nested" / "scan.h5"
    cloud = next(iter(Document.load_hdf5(output).scene_items.values()))
    np.testing.assert_allclose(cloud.points.mean(axis=0), 0, atol=1e-8)
    assert cloud.field_names == ["intensity"]
    np.testing.assert_array_equal(cloud["intensity"].ravel(), [0, 1])
    assert json.loads(sidecar.read_text())["scans"][0]["nonfinite_points_removed"] == 1


@pytest.mark.parametrize("distance", [0., -1., float("nan"), float("inf")])
def test_invalid_distance(tmp_path, distance):
    with pytest.raises(ValueError, match="finite and positive"):
        importer.import_folder(tmp_path, min_distance=distance)


def test_las_dark_colors_are_not_contrast_stretched():
    np.testing.assert_array_equal(importer.rgb_uint8(np.array([[0, 128, 255]]), 65535), [[0, 0, 1]])


def test_same_stem_collision_is_rejected_before_writing(tmp_path):
    (tmp_path / "scan.e57").touch()
    (tmp_path / "scan.las").touch()
    with pytest.raises(ValueError, match="conflicting output names"):
        importer.import_folder(tmp_path)
    assert not list(tmp_path.glob("*.h5"))


def test_e57_ids_restart_for_each_file(tmp_path, monkeypatch):
    for name in ("first.e57", "second.e57"):
        (tmp_path / name).touch()
    monkeypatch.setattr(importer, "read_scans", lambda path: iter([
        (0, np.array([[1000., 0., 0.]]), None, None),
        (1, np.array([[1001., 0., 0.]]), None, None),
    ]))
    results = importer.import_folder(tmp_path)
    assert len(results) == 2
    for output, sidecar in results:
        cloud = next(iter(Document.load_hdf5(output).scene_items.values()))
        assert cloud.field_names == ["scan_id"]
        np.testing.assert_array_equal(cloud["scan_id"].ravel(), [0, 1])
        np.testing.assert_allclose(cloud.points.mean(axis=0), 0)
        assert [scan["scan_id"] for scan in json.loads(sidecar.read_text())["scans"]] == [0, 1]


@pytest.mark.parametrize("values", [
    np.array([0, 100, 200, 300, 65535], dtype=np.uint16),
    np.array([-100., -10., 0., 10., 200.]),
    np.full(5, 42, dtype=np.uint16),
    np.array([0., .2, .5, .8, 1.], dtype=np.float32),
    np.array([], dtype=np.float32),
])
def test_intensity_normalization_matches_reference(values):
    cloud = PointCloudData(np.zeros((len(values), 3), dtype=np.float32))
    cloud.add_field("intensity", values.reshape(-1, 1).copy(), FieldType.INTENSITY)
    expected = values.copy()
    applied = values.size > 0 and (values.max() > 1.0 or values.min() < 0.0)
    if applied:
        expected = expected.astype(np.float64)
        expected -= np.quantile(expected, .025)
        scale = np.quantile(expected, .975)
        if scale > 0:
            expected /= scale
        expected = np.clip(expected, 0, 1)
    result = importer.normalize_cloud_intensity(cloud)
    np.testing.assert_allclose(cloud["intensity"].ravel(), expected)
    assert cloud["intensity"].dtype == expected.dtype
    assert result["applied"] == applied


def test_normalization_excludes_missing_intensity():
    cloud = PointCloudData(np.zeros((5, 3), dtype=np.float32))
    cloud.add_field("intensity", np.array([[10.], [20.], [30.], [0.], [0.]]), FieldType.INTENSITY)
    cloud.add_field("intensity_valid", np.array([[1], [1], [1], [0], [0]], dtype=np.uint8))
    result = importer.normalize_cloud_intensity(cloud)
    np.testing.assert_allclose(cloud["intensity"].ravel(), [0, .5, 1, 0, 0])
    assert result["offset"] == 10.5
    assert result["scale"] == 19


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_normalization_rejects_nonfinite_intensity(bad):
    cloud = PointCloudData(np.zeros((1, 3), dtype=np.float32))
    cloud.add_field("intensity", np.array([[bad]]), FieldType.INTENSITY)
    with pytest.raises(ValueError, match="NaN or infinity"):
        importer.normalize_cloud_intensity(cloud)


@pytest.mark.parametrize("args, expected", [([], True), (["--no-normalize-intensity"], False)])
def test_cli_normalization_switch(monkeypatch, args, expected):
    import sys
    calls = []
    monkeypatch.setattr(sys, "argv", ["import_laser_scans.py", "scans", *args])
    monkeypatch.setattr(importer, "import_folder", lambda *a, **kw: calls.append(kw))
    importer.main()
    assert calls[0]["normalize_intensity"] is expected
