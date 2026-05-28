from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from geon.data.document import Document  # noqa: E402


def _default_output_path(input_path: Path, version: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_format_v{version}{input_path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite a GEON HDF5 document with an unsafe format-version override. "
            "This only changes the written format metadata; it does not migrate data."
        )
    )
    parser.add_argument("input", type=Path, help="Input .h5 GEON document.")
    parser.add_argument("version", type=int, help="Format version to write.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .h5 file. Defaults to '<input>_format_v<VERSION>.h5'.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input file. Use only after making a backup.",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        parser.error(f"Input file does not exist: {input_path}")
    if args.version < 0:
        parser.error("Version must be non-negative.")
    if args.output is not None and args.in_place:
        parser.error("Use either --output or --in-place, not both.")

    output_path = input_path if args.in_place else (args.output or _default_output_path(input_path, args.version))

    doc = Document.load_hdf5(input_path)
    doc.save_hdf5(output_path, unsafe_format_version_override=args.version)
    print(f"Saved {output_path} with geon_format_version={args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
