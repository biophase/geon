from pathlib import Path

from geon.settings import Preferences


def test_preferences_region_growing_round_trip(tmp_path: Path):
    prefs_path = tmp_path / "prefs.toml"
    prefs = Preferences(path=prefs_path)
    prefs.set_region_growing_settings(
        {
            "epsilon": 0.04,
            "tau": 120,
            "enable_chunking": True,
            "normal_mode": "compute",
            "global_reassign_enabled": False,
        }
    )
    prefs.save()

    loaded = Preferences.load(prefs_path)
    rg = loaded.get_region_growing_settings()
    assert rg["epsilon"] == 0.04
    assert rg["tau"] == 120
    assert rg["enable_chunking"] is True
    assert rg["normal_mode"] == "compute"
    assert rg["global_reassign_enabled"] is False


def test_preferences_load_missing_region_growing_keys(tmp_path: Path):
    prefs_path = tmp_path / "prefs.toml"
    prefs_path.write_text(
        'user_name = "Test"\n'
        "enable_telemetry = false\n"
        "camera_sensitivity = 7.5\n",
        encoding="utf-8",
    )

    loaded = Preferences.load(prefs_path)
    assert loaded.user_name == "Test"
    assert loaded.camera_sensitivity == 7.5
    assert loaded.get_region_growing_settings() == {}


def test_preferences_other_segmentation_round_trip(tmp_path: Path):
    prefs_path = tmp_path / "prefs.toml"
    prefs = Preferences(path=prefs_path)
    prefs.set_plane_ransac_settings(
        {
            "epsilon": 0.05,
            "normal_mode": "compute",
            "output_field_base": "planes",
        }
    )
    prefs.set_superpoints_settings(
        {
            "k_neighbors": 12,
            "parallel": True,
            "feature_field_names": ["normals", "intensity"],
        }
    )
    prefs.set_region_merge_settings(
        {
            "neighbor_radius": 0.08,
            "source_field_name": "superpoints",
            "min_region_size": 30,
        }
    )
    prefs.save()

    loaded = Preferences.load(prefs_path)
    assert loaded.get_plane_ransac_settings()["epsilon"] == 0.05
    assert loaded.get_plane_ransac_settings()["normal_mode"] == "compute"
    assert loaded.get_superpoints_settings()["feature_field_names"] == ["normals", "intensity"]
    assert loaded.get_region_merge_settings()["source_field_name"] == "superpoints"
