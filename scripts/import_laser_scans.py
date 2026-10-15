"""Convert each E57/LAS/LAZ file into its own centered Geon document.

Run in geon_dev (install readers with: python -m pip install pye57 "laspy[lazrs]"):
    python scripts/import_laser_scans.py D:/scans --output-dir D:/output

Coordinates must be in metres. E57 scan poses are
applied; LAS scale/offset are decoded by laspy. No registration or reprojection
is performed. Scan positions within each E57 file are merged and get a
zero-based scan_id. LAS/LAZ outputs do not have a scan_id field. Each input
is centered and subsampled independently, producing <stem>.h5 and <stem>.json.
Outputs default to beside each input; --output-dir preserves relative subfolders.
The default spatial subsampling distance is 0.01 m: minimum retained spacing,
not a guarantee that gaps in the source cloud are at most 1 cm.

The JSON sidecar records the pre-subsampling centroid and both translations.
Centering uses float64 before casting to float32; subsampling can shift the
remaining points' mean. One input file is processed in memory at a time. Sources are read
only. RGB is uint8; missing colors are black with rgb_valid=0 in mixed clouds.
After subsampling, intensities outside [0, 1] are normalized in float64 using
the 2.5th/97.5th percentiles and clipped to [0, 1]. Values already in [0, 1]
are unchanged. Use --no-normalize-intensity to retain reader values and dtype
(E57 float32, LAS/LAZ uint16). Missing intensity is filled with zero and
marked intensity_valid=0; no intensity field is added if all scans lack it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from geon.algorithms.subsampling import spatial_subsample_mask
from geon.data.document import Document
from geon.data.pointcloud import FieldType, PointCloudData


def rgb_uint8(rgb: np.ndarray, maximum: float) -> np.ndarray:
    """Scale a known source color range without per-scan contrast stretching."""
    values = np.asarray(rgb, dtype=np.float64)
    return np.rint(np.clip(values, 0, maximum) * (255.0 / maximum)).astype(np.uint8)


def normalize_cloud_intensity(cloud: PointCloudData) -> dict:
    """Normalize retained, present intensities using the reference quantiles."""
    result = {"applied": False}
    fields = cloud.get_fields(field_type=FieldType.INTENSITY)
    if not fields:
        return result
    field = fields[0]
    present = (cloud["intensity_valid"].ravel().astype(bool)
               if "intensity_valid" in cloud.field_names
               else np.ones(len(cloud.points), dtype=bool))
    values = field.data[present]
    if values.size == 0:
        return result
    if not np.isfinite(values).all():
        raise ValueError("Intensity contains NaN or infinity; cannot normalize")
    if values.min() >= 0.0 and values.max() <= 1.0:
        return result
    values = values.astype(np.float64, copy=True)
    offset = float(np.quantile(values, 0.025))
    values -= offset
    scale = float(np.quantile(values, 0.975))
    if scale > 0:
        values /= scale
    normalized = field.data.astype(np.float64, copy=True)
    normalized[present] = np.clip(values, 0, 1)
    field.data = normalized
    return {"applied": True, "offset": offset, "scale": scale}


def read_scans(path: Path) -> Iterator[tuple[int, np.ndarray, np.ndarray | None, np.ndarray | None]]:
    if path.suffix.lower() == ".e57":
        try:
            import pye57
        except ImportError as exc:
            raise RuntimeError("E57 support requires: python -m pip install pye57") from exc
        with pye57.E57(str(path), mode="r") as reader:
            for index in range(reader.scan_count):
                # read_scan filters invalid returns, converts spherical scans,
                # and applies the scan's pose in float64.
                data = reader.read_scan(
                    index, colors=True, intensity=True, transform=True, ignore_missing_fields=True
                )
                points = np.column_stack([data[f"cartesian{axis}"] for axis in "XYZ"])
                keys = ["colorRed", "colorGreen", "colorBlue"]
                rgb = None
                if all(key in data for key in keys):
                    # pye57 exposes E57 color channels as uint8.
                    rgb = rgb_uint8(np.column_stack([data[key] for key in keys]), 255)
                yield index, np.asarray(points, dtype=np.float64), rgb, data.get("intensity")
    else:
        try:
            import laspy
        except ImportError as exc:
            raise RuntimeError('LAS/LAZ support requires: python -m pip install "laspy[lazrs]"') from exc
        las = laspy.read(path)
        points = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)
        rgb = None
        if {"red", "green", "blue"}.issubset(las.point_format.dimension_names):
            # LAS specifies 16-bit RGB, including for very dark scans.
            rgb = rgb_uint8(np.column_stack((las.red, las.green, las.blue)), 65535)
        yield 0, points, rgb, np.asarray(las.intensity)


def import_folder(
    folder: Path,
    output_dir: Path | None = None,
    *,
    recursive: bool = False,
    min_distance: float = 0.01,
    normalize_intensity: bool = True,
) -> list[tuple[Path, Path]]:
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"Input folder does not exist: {folder}")
    if not np.isfinite(min_distance) or min_distance <= 0:
        raise ValueError("min_distance must be finite and positive")
    files = sorted(
        (p for p in folder.glob("**/*" if recursive else "*")
         if p.is_file() and p.suffix.lower() in {".e57", ".las", ".laz"}),
        key=lambda p: (str(p).casefold(), str(p)),
    )
    if not files:
        raise ValueError(f"No E57, LAS or LAZ files found in {folder}")
    output_dir = output_dir.resolve() if output_dir is not None else folder
    if output_dir.is_file() or output_dir.suffix.lower() in {".h5", ".hdf5"}:
        raise ValueError("Output must be a directory; each input gets its own .h5 file")
    jobs = [(path, (output_dir / path.relative_to(folder)).with_suffix(".h5")) for path in files]
    # Preflight the whole batch, including same-stem inputs such as a.las/a.e57.
    destinations = set()
    for _, output in jobs:
        for destination in (output, output.with_suffix(".json")):
            key = str(destination).casefold()
            if key in destinations:
                raise ValueError(f"Inputs have conflicting output names: {destination}; rename one input")
            destinations.add(key)
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {destination}")
    return [import_file(path, output, min_distance=min_distance,
                        normalize_intensity=normalize_intensity) for path, output in jobs]


def import_file(path: Path, output: Path, *, min_distance: float = 0.01,
                normalize_intensity: bool = True) -> tuple[Path, Path]:
    """Center and subsample one file; retain scan IDs only for E57 positions."""
    path, output = path.resolve(), output.resolve()
    if not np.isfinite(min_distance) or min_distance <= 0:
        raise ValueError("min_distance must be finite and positive")
    if path.suffix.lower() not in {".e57", ".las", ".laz"}:
        raise ValueError(f"Unsupported input file: {path}")
    is_e57 = path.suffix.lower() == ".e57"
    if output.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("Output must have an .h5 or .hdf5 extension")
    sidecar = output.with_suffix(".json")
    for destination in (output, sidecar):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {destination}")

    points_parts = []
    rgb_parts = []
    intensity_parts = []
    id_parts = []
    scans = []
    print(f"Reading {path}", flush=True)
    for source_index, points, rgb, intensity in read_scans(path):
        finite = np.isfinite(points).all(axis=1)
        valid_points = points[finite]
        count = len(valid_points)
        scans.append({
            **({"scan_id": source_index, "source_scan_index": source_index} if is_e57 else {}),
            "source_file": str(path),
            "point_count_before_subsampling": count,
            "nonfinite_points_removed": int(len(points) - count),
            "has_rgb": rgb is not None,
            "has_intensity": intensity is not None,
        })
        points_parts.append(valid_points)
        rgb_parts.append(rgb[finite] if rgb is not None else None)
        intensity_parts.append(intensity[finite].reshape(-1, 1) if intensity is not None else None)
        if is_e57:
            id_parts.append(np.full((count, 1), source_index, dtype=np.uint32))
    if not points_parts or not any(len(part) for part in points_parts):
        raise ValueError("Input scans contain no valid points")

    points = np.concatenate(points_parts)
    points_parts.clear()
    mean = points.mean(axis=0, dtype=np.float64)
    points -= mean
    cloud = PointCloudData(points.astype(np.float32))
    del points
    if not np.isfinite(cloud.points).all():
        raise ValueError("Centered coordinates exceed the float32 range")
    if is_e57:
        cloud.add_field("scan_id", np.concatenate(id_parts), FieldType.SCALAR)
    if any(part is not None for part in rgb_parts):
        colors = np.concatenate([
            part if part is not None else np.zeros((scan["point_count_before_subsampling"], 3), dtype=np.uint8)
            for part, scan in zip(rgb_parts, scans)
        ])
        cloud.add_field("color", colors, FieldType.COLOR)
        if any(part is None for part in rgb_parts):
            valid_rgb = np.concatenate([
                np.full((scan["point_count_before_subsampling"], 1), part is not None, dtype=np.uint8)
                for part, scan in zip(rgb_parts, scans)
            ])
            cloud.add_field("rgb_valid", valid_rgb, FieldType.SCALAR)
    rgb_parts.clear()
    if any(part is not None for part in intensity_parts):
        intensity_dtype = np.result_type(*[part.dtype for part in intensity_parts if part is not None])
        intensities = np.concatenate([
            part if part is not None else np.zeros(
                (scan["point_count_before_subsampling"], 1), dtype=intensity_dtype
            )
            for part, scan in zip(intensity_parts, scans)
        ])
        cloud.add_field("intensity", intensities, FieldType.INTENSITY)
        if any(part is None for part in intensity_parts):
            valid_intensity = np.concatenate([
                np.full((scan["point_count_before_subsampling"], 1), part is not None, dtype=np.uint8)
                for part, scan in zip(intensity_parts, scans)
            ])
            cloud.add_field("intensity_valid", valid_intensity, FieldType.SCALAR)
    intensity_parts.clear()
    id_parts.clear()

    before = len(cloud.points)
    print(f"Subsampling {before:,} points at {min_distance:g} m", flush=True)
    mask = spatial_subsample_mask(cloud.points, min_distance)
    # Apply the same selection to every field, retaining its dtype.
    cloud.points = cloud.points[mask]
    for field in cloud.get_fields():
        field.data = field.data[mask]
    counts = (np.bincount(cloud["scan_id"].ravel(), minlength=len(scans))
              if is_e57 else [len(cloud.points)])
    for scan, count in zip(scans, counts):
        scan["point_count_after_subsampling"] = int(count)

    intensity_normalization = {
        "enabled": normalize_intensity,
        "stage": "after subsampling; excludes missing intensity placeholders",
        "quantiles": [0.025, 0.975],
        **(normalize_cloud_intensity(cloud) if normalize_intensity else {"applied": False}),
    }

    metadata = {
        "units": "metres",
        "coordinate_convention": "local = world + world_to_local_translation",
        "world_mean_before_subsampling": mean.tolist(),
        "world_to_local_translation": (-mean).tolist(),
        "local_to_world_translation": mean.tolist(),
        "coordinate_dtype": "float32",
        "centering_stage": "before subsampling; retained mean may differ from zero",
        "subsampling": {"algorithm": "spatial_subsample_mask", "min_distance_m": min_distance},
        "point_count_before_subsampling": before,
        "point_count_after_subsampling": len(cloud.points),
        "rgb_encoding": "uint8 0-255; LAS RGB scaled from 0-65535; missing RGB=0, rgb_valid=0",
        "intensity_encoding": (
            "float64 normalized to [0, 1]" if intensity_normalization["applied"]
            else "reader values and dtype unchanged"
        ) + "; missing intensity=0, intensity_valid=0",
        "intensity_normalization": intensity_normalization,
        "scans": scans,
    }
    metadata_json = json.dumps(metadata, indent=2, allow_nan=False)
    doc = Document(path.stem)
    doc.add_data(cloud)
    # Keep the transform recoverable even when the sidecar is separated.
    doc.meta["laser_scan_import"] = metadata_json
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save_hdf5(output)
    sidecar.write_text(metadata_json + "\n", encoding="utf-8")
    print(f"Saved {len(cloud.points):,} points to {output}\nTranslation: {sidecar}")
    return output, sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Folder containing E57/LAS/LAZ files in metres")
    parser.add_argument("--output-dir", "--output", "-o", type=Path,
                        help="Output directory (default: beside each input); one .h5/.json pair per file")
    parser.add_argument("--recursive", action="store_true", help="Also discover files in subfolders")
    parser.add_argument("--min-distance", type=float, default=0.01, help="Minimum retained spacing in metres (default: 0.01)")
    parser.add_argument("--no-normalize-intensity", dest="normalize_intensity", action="store_false",
                        help="Keep original intensity values and dtype instead of percentile normalization")
    args = parser.parse_args()
    try:
        import_folder(args.folder, args.output_dir, recursive=args.recursive, min_distance=args.min_distance,
                      normalize_intensity=args.normalize_intensity)
    except (ValueError, FileExistsError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")


if __name__ == "__main__":
    main()
