from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore

from geon.algorithms.region_growing import estimate_parameters, segment_planar_regions
from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData
from geon.settings import Preferences


DEFAULT_REGION_GROWING_SETTINGS: dict[str, object] = {
    "epsilon": 0.03,
    "tau": 80,
    "alpha_deg": 29.0,
    "normal_mode": "compute",
    "normal_field_name": None,
    "output_field_base": "planar_regions",
    "confidence": 0.99,
    "enable_seed_gating": True,
    "seed_min_neighbors": 10,
    "seed_planarity_min": 0.20,
    "seed_scattering_max": 0.35,
    "failrate_window": 64,
    "failrate_threshold": 0.90,
    "chunk_mode": "auto",
    "enable_chunking": True,
    "target_points_per_chunk": 250_000,
    "chunk_x": 2,
    "chunk_y": 2,
    "chunk_z": 1,
    "overlap_factor": 3.0,
    "merge_angle_deg": 5.0,
    "merge_distance_factor": 3.0,
    "enable_reconciliation": True,
    "epsilon_multiplier": 3.0,
    "refit_multiplier": 2.0,
    "first_refit": 4,
    "max_dist_from_cent": 50.0,
    "oriented_normals": False,
    "perform_cca": True,
    "local_reassign_enabled": True,
    "global_reassign_enabled": True,
}


def _unique_field_name(existing: list[str], base: str) -> str:
    if base not in existing:
        return base
    suffix = 1
    while True:
        candidate = f"{base}_{suffix:03d}"
        if candidate not in existing:
            return candidate
        suffix += 1


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() in {".toml", ".tml"}:
        if tomllib is None:
            raise RuntimeError("TOML config requires Python 3.11+ or tomllib availability.")
        return tomllib.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported config file type: {path.suffix}")


def _extract_region_growing_settings(raw: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def _pull(src: dict[str, Any], mapping: dict[str, str] | None = None) -> None:
        if mapping is None:
            mapping = {k: k for k in DEFAULT_REGION_GROWING_SETTINGS}
        for src_key, dst_key in mapping.items():
            if src_key in src:
                out[dst_key] = src[src_key]

    _pull(raw)

    rg = raw.get("region_growing")
    if isinstance(rg, dict):
        _pull(rg)
        params = rg.get("params")
        if isinstance(params, dict):
            _pull(params)
        chunking = rg.get("chunking")
        if isinstance(chunking, dict):
            _pull(chunking, {
                "enabled": "enable_chunking",
                "mode": "chunk_mode",
                "target_points_per_chunk": "target_points_per_chunk",
                "chunk_x": "chunk_x",
                "chunk_y": "chunk_y",
                "chunk_z": "chunk_z",
                "overlap_factor": "overlap_factor",
            })
        merge = rg.get("merge")
        if isinstance(merge, dict):
            _pull(merge, {
                "enabled": "enable_reconciliation",
                "angle_deg": "merge_angle_deg",
                "distance_factor": "merge_distance_factor",
            })

    return out


def _build_params(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "epsilon": float(settings["epsilon"]),
        "tau": int(settings["tau"]),
        "alpha_deg": float(settings["alpha_deg"]),
        "confidence": float(settings["confidence"]),
        "epsilon_multiplier": float(settings["epsilon_multiplier"]),
        "refit_multiplier": float(settings["refit_multiplier"]),
        "first_refit": int(settings["first_refit"]),
        "max_dist_from_cent": float(settings["max_dist_from_cent"]),
        "oriented_normals": bool(settings["oriented_normals"]),
        "perform_cca": bool(settings["perform_cca"]),
        "refine_unassigned": bool(settings["global_reassign_enabled"]),
        "local_reassign_enabled": bool(settings["local_reassign_enabled"]),
        "global_reassign_enabled": bool(settings["global_reassign_enabled"]),
        "enable_seed_gating": bool(settings["enable_seed_gating"]),
        "seed_min_neighbors": int(settings["seed_min_neighbors"]),
        "seed_planarity_min": float(settings["seed_planarity_min"]),
        "seed_scattering_max": float(settings["seed_scattering_max"]),
        "failrate_window": int(settings["failrate_window"]),
        "failrate_threshold": float(settings["failrate_threshold"]),
    }


def _build_chunking(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(settings["enable_chunking"]),
        "mode": str(settings["chunk_mode"]),
        "target_points_per_chunk": int(settings["target_points_per_chunk"]),
        "chunk_x": int(settings["chunk_x"]),
        "chunk_y": int(settings["chunk_y"]),
        "chunk_z": int(settings["chunk_z"]),
        "overlap_factor": float(settings["overlap_factor"]),
    }


def _build_merge(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(settings["enable_reconciliation"]),
        "angle_deg": float(settings["merge_angle_deg"]),
        "distance_factor": float(settings["merge_distance_factor"]),
    }


def _resolve_normals(data: PointCloudData, settings: dict[str, Any]) -> tuple[np.ndarray | None, str]:
    normal_mode = str(settings.get("normal_mode") or "compute")
    if normal_mode != "use_provided":
        return None, "compute"

    field_name = settings.get("normal_field_name")
    normal_fields = []
    if isinstance(field_name, str) and field_name.strip():
        normal_fields = data.get_fields(names=field_name.strip(), field_type=FieldType.NORMAL)
        if not normal_fields:
            raise RuntimeError(f"Normals field '{field_name}' was not found.")
    else:
        normal_fields = data.get_fields(field_type=FieldType.NORMAL)
        if not normal_fields:
            raise RuntimeError("normal_mode=use_provided but no NORMAL field exists.")

    normals = np.asarray(normal_fields[0].data, dtype=np.float32)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise RuntimeError(f"Normals field '{normal_fields[0].name}' must have shape (N,3).")
    return normals, "use_provided"


def _iter_h5_files(dataset_folder: Path) -> list[Path]:
    files = sorted(dataset_folder.rglob("*.h5"))
    files.extend(sorted(dataset_folder.rglob("*.hdf5")))
    deduped: dict[str, Path] = {}
    for fp in files:
        deduped[str(fp.resolve())] = fp
    return list(deduped.values())


def _process_document(path: Path, settings: dict[str, Any], *, estimate: bool) -> int:
    doc = Document.load_hdf5(path)
    output_base = str(settings["output_field_base"]).strip() or "planar_regions"
    point_cloud_item: tuple[str, PointCloudData] | None = None
    for item_name, item in doc.scene_items.items():
        if isinstance(item, PointCloudData):
            point_cloud_item = (item_name, item)
            break

    if point_cloud_item is None:
        print("  no PointCloudData items found, skipping save")
        return 0

    item_name, item = point_cloud_item
    run_settings = dict(settings)
    if estimate:
        estimated = estimate_parameters(item)
        run_settings["epsilon"] = float(estimated["epsilon"])
        run_settings["tau"] = int(estimated["tau"])
        run_settings["alpha_deg"] = float(estimated["alpha_deg"])
        print(
            "  estimated core params: "
            f"epsilon={run_settings['epsilon']:.6f} "
            f"tau={run_settings['tau']} "
            f"alpha_deg={run_settings['alpha_deg']:.3f}"
        )

    params = _build_params(run_settings)
    chunking = _build_chunking(run_settings)
    merge = _build_merge(run_settings)
    normals, normal_mode = _resolve_normals(item, run_settings)
    field_name = _unique_field_name(item.field_names, output_base)
    print(
        f"  point cloud '{item_name}': {item.points.shape[0]:,} pts | "
        f"normal_mode={normal_mode} | output={field_name}"
    )
    labels, stats = segment_planar_regions(
        item,
        normals=normals,
        normal_mode=normal_mode,
        params=params,
        chunking=chunking,
        merge=merge,
    )
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    item.add_field(
        name=field_name,
        data=labels[:, None],
        field_type=FieldType.INSTANCE,
    )
    print(
        f"    regions={int(stats.get('num_regions_post_merge', -1))} | "
        f"unassigned={int(stats.get('num_unassigned', -1))} | "
        f"elapsed={float(stats.get('elapsed_seconds', 0.0)):.3f}s"
    )

    doc.save_hdf5(path)
    return 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch planar region growing over all .h5/.hdf5 documents in a dataset folder.",
    )
    parser.add_argument("dataset_folder", help="Dataset root folder containing GEON .h5/.hdf5 documents.")
    parser.add_argument("--config", type=Path, help="Optional TOML or JSON config file.")
    parser.add_argument("--estimate", action="store_true", help="Run heuristic parameter estimation per document.")
    parser.add_argument("--output-field-base")
    parser.add_argument("--normal-mode", choices=["compute", "use_provided"])
    parser.add_argument("--normal-field-name")
    parser.add_argument("--chunk-mode", choices=["auto", "explicit"])
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--tau", type=int)
    parser.add_argument("--alpha-deg", dest="alpha_deg", type=float)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--seed-min-neighbors", dest="seed_min_neighbors", type=int)
    parser.add_argument("--seed-planarity-min", dest="seed_planarity_min", type=float)
    parser.add_argument("--seed-scattering-max", dest="seed_scattering_max", type=float)
    parser.add_argument("--failrate-window", dest="failrate_window", type=int)
    parser.add_argument("--failrate-threshold", dest="failrate_threshold", type=float)
    parser.add_argument("--target-points-per-chunk", dest="target_points_per_chunk", type=int)
    parser.add_argument("--chunk-x", dest="chunk_x", type=int)
    parser.add_argument("--chunk-y", dest="chunk_y", type=int)
    parser.add_argument("--chunk-z", dest="chunk_z", type=int)
    parser.add_argument("--overlap-factor", dest="overlap_factor", type=float)
    parser.add_argument("--merge-angle-deg", dest="merge_angle_deg", type=float)
    parser.add_argument("--merge-distance-factor", dest="merge_distance_factor", type=float)
    parser.add_argument("--epsilon-multiplier", dest="epsilon_multiplier", type=float)
    parser.add_argument("--refit-multiplier", dest="refit_multiplier", type=float)
    parser.add_argument("--first-refit", dest="first_refit", type=int)
    parser.add_argument("--max-dist-from-cent", dest="max_dist_from_cent", type=float)

    bool_action = argparse.BooleanOptionalAction
    parser.add_argument("--enable-seed-gating", dest="enable_seed_gating", action=bool_action, default=None)
    parser.add_argument("--enable-chunking", dest="enable_chunking", action=bool_action, default=None)
    parser.add_argument("--enable-reconciliation", dest="enable_reconciliation", action=bool_action, default=None)
    parser.add_argument("--oriented-normals", dest="oriented_normals", action=bool_action, default=None)
    parser.add_argument("--perform-cca", dest="perform_cca", action=bool_action, default=None)
    parser.add_argument("--local-reassign-enabled", dest="local_reassign_enabled", action=bool_action, default=None)
    parser.add_argument("--global-reassign-enabled", dest="global_reassign_enabled", action=bool_action, default=None)
    parser.add_argument("--continue-on-error", action=bool_action, default=True)
    return parser


def _merged_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = dict(DEFAULT_REGION_GROWING_SETTINGS)
    settings.update(Preferences.load().get_region_growing_settings())

    if args.config is not None:
        settings.update(_extract_region_growing_settings(_load_config(args.config)))

    for key in DEFAULT_REGION_GROWING_SETTINGS:
        value = getattr(args, key, None)
        if value is not None:
            settings[key] = value

    return settings


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    dataset_folder = Path(args.dataset_folder).expanduser().resolve()
    if not dataset_folder.exists() or not dataset_folder.is_dir():
        parser.error(f"dataset_folder does not exist or is not a directory: {dataset_folder}")

    settings = _merged_settings(args)
    files = _iter_h5_files(dataset_folder)
    if not files:
        print(f"No .h5/.hdf5 files found under {dataset_folder}")
        return 1

    print(f"Dataset folder: {dataset_folder}")
    print(f"Documents found: {len(files)}")
    print(
        "Region growing settings: "
        f"epsilon={settings['epsilon']} tau={settings['tau']} alpha_deg={settings['alpha_deg']} "
        f"normal_mode={settings['normal_mode']} chunking={settings['enable_chunking']} "
        f"local_reassign={settings['local_reassign_enabled']} "
        f"global_reassign={settings['global_reassign_enabled']}"
    )

    processed = 0
    failures = 0
    for i, fp in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {fp}")
        try:
            changed_fields = _process_document(fp, settings, estimate=bool(args.estimate))
            processed += 1
            print(f"  saved {changed_fields} new instance field(s)")
        except Exception as exc:
            failures += 1
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            if not args.continue_on_error:
                break

    print(f"Finished: processed={processed} failures={failures}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
