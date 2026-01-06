from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData


NPY_ROOT = Path("data/test_dataset/labeled-snrw-npy")
NPY_GLOB = "*.npy"
PLY_ROOT = Path("data/_temp/bw52-130")
PLY_GLOB = "**/preprocessed/*.ply"
OUTPUT_PATH = Path("data/_temp/bw52-130/npy_to_ply_map.txt")
OUTPUT_DUPES_PATH = Path("data/_temp/bw52-130/npy_to_ply_map_duplicates.txt")


def _load_npy_shape(path: Path) -> tuple[int, ...]:
    arr = np.load(path)
    return tuple(arr.shape)


def _load_ply_vertex_count(path: Path) -> int:
    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY missing vertex element: {path}")
    return len(ply["vertex"].data)


def main() -> None:
    npy_paths = sorted(NPY_ROOT.glob(NPY_GLOB))
    if not npy_paths:
        raise SystemExit(f"No npy files found under {NPY_ROOT}/{NPY_GLOB}")

    ply_paths = sorted(PLY_ROOT.glob(PLY_GLOB))
    if not ply_paths:
        raise SystemExit(f"No ply files found under {PLY_ROOT}/{PLY_GLOB}")

    npy_shapes: dict[Path, tuple[int, ...]] = {}
    print("NPY shapes:")
    for npy_path in npy_paths:
        shape = _load_npy_shape(npy_path)
        npy_shapes[npy_path] = shape
        print(f"  {npy_path.name}: {shape}")

    ply_vertex_counts: dict[Path, int] = {}
    print("\nPLY vertex counts:")
    for ply_path in ply_paths:
        count = _load_ply_vertex_count(ply_path)
        ply_vertex_counts[ply_path] = count
        print(f"  {ply_path.name}: {count}")

    output_lines: list[str] = []
    duplicate_lines: list[str] = []
    print("\nNPY -> PLY matches:")
    for npy_path in npy_paths:
        shape = npy_shapes[npy_path]
        if not shape:
            print(f"  {npy_path.name}: empty shape, skipping")
            continue
        target_count = shape[0]
        matches = [
            ply_path
            for ply_path, count in ply_vertex_counts.items()
            if count == target_count
        ]
        if len(matches) == 1:
            ply_path = matches[0]
            output_lines.append(f"{npy_path.name}\t{ply_path.name}")
            print(f"  {npy_path.name} -> {ply_path.name}")
        elif not matches:
            print(f"  {npy_path.name}: no ply match for {target_count} points")
        else:
            match_names = ", ".join(sorted(p.name for p in matches))
            duplicate_lines.append(f"{npy_path.name}\t{match_names}")
            print(
                f"  {npy_path.name}: {len(matches)} ply matches for {target_count} points"
            )

    if output_lines:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text("\n".join(output_lines) + "\n")
        print(f"\nWrote mapping to {OUTPUT_PATH}")
    else:
        print("\nNo unique matches found; output file not written.")

    if duplicate_lines:
        OUTPUT_DUPES_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_DUPES_PATH.write_text("\n".join(duplicate_lines) + "\n")
        print(f"Wrote duplicate matches to {OUTPUT_DUPES_PATH}")
    else:
        print("No duplicate matches found.")


if __name__ == "__main__":
    main()
