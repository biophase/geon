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
            "on_selection_only": True,
            "output_mode": "write_existing",
            "output_existing_field_name": "instances",
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
    assert rg["on_selection_only"] is True
    assert rg["output_mode"] == "write_existing"
    assert rg["output_existing_field_name"] == "instances"


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


def test_preferences_cell_complex_round_trip(tmp_path: Path):
    prefs_path = tmp_path / "prefs.toml"
    prefs = Preferences(path=prefs_path)
    prefs.cell_complex_size_mode = "world"
    prefs.cell_complex_screen_size_px = 18.0
    prefs.cell_complex_world_size = 0.25
    prefs.cell_complex_edge_width = 3.0
    prefs.cell_complex_default_color = [1, 2, 3]
    prefs.viewport_text_color = [4, 5, 6]
    prefs.save()

    loaded = Preferences.load(prefs_path)
    assert loaded.cell_complex_size_mode == "world"
    assert loaded.cell_complex_screen_size_px == 18.0
    assert loaded.cell_complex_world_size == 0.25
    assert loaded.cell_complex_edge_width == 3.0
    assert loaded.cell_complex_default_color == [1, 2, 3]
    assert loaded.viewport_text_color == [4, 5, 6]


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
    prefs.set_corner_cleanup_settings(
        {
            "source_field_name": "regions",
            "on_selection_only": True,
            "neighbor_radius_factor": 3.0,
        }
    )
    prefs.set_connected_components_settings(
        {
            "epsilon": 0.07,
            "on_selection_only": True,
            "output_mode": "write_existing",
            "output_existing_field_name": "regions",
        }
    )
    prefs.save()

    loaded = Preferences.load(prefs_path)
    assert loaded.get_plane_ransac_settings()["epsilon"] == 0.05
    assert loaded.get_plane_ransac_settings()["normal_mode"] == "compute"
    assert loaded.get_superpoints_settings()["feature_field_names"] == ["normals", "intensity"]
    assert loaded.get_region_merge_settings()["source_field_name"] == "superpoints"
    corner_cleanup = loaded.get_corner_cleanup_settings()
    assert corner_cleanup["source_field_name"] == "regions"
    assert corner_cleanup["on_selection_only"] is True
    assert corner_cleanup["neighbor_radius_factor"] == 3.0
    connected = loaded.get_connected_components_settings()
    assert connected["epsilon"] == 0.07
    assert connected["on_selection_only"] is True
    assert connected["output_mode"] == "write_existing"
    assert connected["output_existing_field_name"] == "regions"


def test_unassigned_color_round_trip(tmp_path):
    prefs = Preferences(path=tmp_path / "prefs.toml")
    prefs.unassigned_point_color = [12, 34, 56, 78]
    prefs.save()
    assert Preferences.load(prefs.path).unassigned_point_color == [12, 34, 56, 78]


def test_unassigned_color_legacy_and_invalid_values(tmp_path):
    path = tmp_path / "prefs.toml"
    cases = [
        ('user_name = "Legacy"', [204, 204, 204, 192]),
        ('unassigned_point_color = [1, 2, 3]', [204, 204, 204, 192]),
        ('unassigned_point_color = "bad"', [204, 204, 204, 192]),
        ('unassigned_point_color = [1, 2, "bad", 4]', [204, 204, 204, 192]),
        ('unassigned_point_color = [1, 2, inf, 4]', [204, 204, 204, 192]),
        ('unassigned_point_color = [-10, 300, 12, 999]', [0, 255, 12, 255]),
    ]
    for text, expected in cases:
        path.write_text(text, encoding="utf-8")
        assert Preferences.load(path).unassigned_point_color == expected
