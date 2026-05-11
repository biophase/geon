from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from glob import glob
import logging
import os.path as osp
from pathlib import Path
import sys
from typing import Dict, Iterable, Literal, Optional, Tuple, cast

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

Color = Tuple[int, int, int]
UnlabeledPolicy = Literal["include", "empty"]
DEFAULT_UNLABELED_COLOR: Color = (204, 204, 204)


def _import_geon_types() -> None:
    global Document
    global PointCloudData
    global SemanticClass
    global SemanticSchema
    global SemanticSegmentation

    from geon.data.document import Document as _Document
    from geon.data.pointcloud import (
        PointCloudData as _PointCloudData,
        SemanticClass as _SemanticClass,
        SemanticSchema as _SemanticSchema,
        SemanticSegmentation as _SemanticSegmentation,
    )

    Document = _Document
    PointCloudData = _PointCloudData
    SemanticClass = _SemanticClass
    SemanticSchema = _SemanticSchema
    SemanticSegmentation = _SemanticSegmentation


@dataclass(frozen=True)
class FieldLocation:
    doc_path: Path
    doc_name: str
    pointcloud_id: str
    field_name: str
    schema_name: str
    schema: SemanticSchema

    @property
    def key(self) -> str:
        return f"{self.doc_name}/{self.pointcloud_id}/{self.field_name}/{self.schema_name}"


def _iter_document_paths(dataset_fp: str) -> list[Path]:
    paths = []
    paths.extend(glob(osp.join(dataset_fp, "*.h5")))
    paths.extend(glob(osp.join(dataset_fp, "*.hdf5")))
    paths.extend(glob(osp.join(dataset_fp, "*", "*.h5")))
    paths.extend(glob(osp.join(dataset_fp, "*", "*.hdf5")))
    return sorted({Path(path) for path in paths})


def _scan_locations(dataset_fp: str, schema_name_filter: Optional[str]) -> list[FieldLocation]:
    locations: list[FieldLocation] = []
    for doc_path in _iter_document_paths(dataset_fp):
        for key, schema in SemanticSchema.scan_h5(doc_path).items():
            doc_name, pointcloud_id, field_name, schema_name = key.split("/", 3)
            if schema_name_filter is not None and schema_name != schema_name_filter:
                continue
            locations.append(
                FieldLocation(
                    doc_path=doc_path,
                    doc_name=doc_name,
                    pointcloud_id=pointcloud_id,
                    field_name=field_name,
                    schema_name=schema_name,
                    schema=schema,
                )
            )
    return locations


def _schema_name_counts(schema: SemanticSchema) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sem_cls in schema.semantic_classes:
        counts[sem_cls.name] = counts.get(sem_cls.name, 0) + 1
    return counts


def _schema_id_to_class(schema: SemanticSchema) -> Dict[int, SemanticClass]:
    return {int(sem_cls.id): sem_cls for sem_cls in schema.semantic_classes}


def _choose_unlabeled(class_colors: Dict[str, Color]) -> SemanticClass:
    if "_unlabeled" in class_colors:
        return SemanticClass(-1, "_unlabeled", class_colors["_unlabeled"])
    return SemanticClass(-1, "_unlabeled", DEFAULT_UNLABELED_COLOR)


def _build_global_schema(
    schema_name: str,
    class_colors: Dict[str, Color],
    unlabeled_policy: UnlabeledPolicy,
) -> SemanticSchema:
    classes: list[SemanticClass] = []
    if unlabeled_policy == "include":
        classes.append(_choose_unlabeled(class_colors))

    class_names = sorted(name for name in class_colors if name != "_unlabeled")
    for class_id, class_name in enumerate(class_names):
        classes.append(SemanticClass(class_id, class_name, class_colors[class_name]))

    return SemanticSchema(name=schema_name, semantic_classes=classes)


def _classes_by_name(schema: SemanticSchema) -> Dict[str, SemanticClass]:
    return {sem_cls.name: sem_cls for sem_cls in schema.semantic_classes}


def _remap_pairs_by_name(source_schema: SemanticSchema, target_schema: SemanticSchema) -> list[tuple[int, int]]:
    target_by_name = _classes_by_name(target_schema)
    pairs: list[tuple[int, int]] = []
    for source_cls in source_schema.semantic_classes:
        if source_cls.name not in target_by_name:
            continue
        target_cls = target_by_name[source_cls.name]
        pairs.append((int(source_cls.id), int(target_cls.id)))
    return pairs


def _ids_not_in_schema(field: SemanticSegmentation) -> list[int]:
    schema_ids = set(_schema_id_to_class(field.schema))
    return sorted(int(uid) for uid in np.unique(field.data) if int(uid) not in schema_ids)


def _same_signature_ignore_schema_name(left: SemanticSchema, right: SemanticSchema) -> bool:
    return left.signature() == right.signature()


def _configure_logging(log_path: Path) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def _warn(message: str) -> None:
    logging.warning("\033[33m%s\033[0m", message)


def _error(message: str) -> None:
    logging.error("\033[31m%s\033[0m", message)


def _audit_schemas(
    locations: Iterable[FieldLocation],
    unlabeled_policy: UnlabeledPolicy,
) -> tuple[dict[str, SemanticSchema], bool]:
    class_colors_by_schema: dict[str, dict[str, Color]] = {}
    class_sources_by_schema: dict[str, dict[str, str]] = {}
    has_errors = False

    for location in locations:
        schema_name = location.schema_name
        class_colors = class_colors_by_schema.setdefault(schema_name, {})
        class_sources = class_sources_by_schema.setdefault(schema_name, {})

        duplicate_names = [
            name for name, count in _schema_name_counts(location.schema).items() if count > 1
        ]
        if duplicate_names:
            _error(
                f"[Audit] {location.key}: duplicate class name(s) in schema: {duplicate_names}. "
                "This schema cannot be safely unified."
            )
            has_errors = True
            continue

        for sem_cls in location.schema.semantic_classes:
            class_name = sem_cls.name
            color = tuple(int(channel) for channel in sem_cls.color)
            if class_name not in class_colors:
                class_colors[class_name] = color
                class_sources[class_name] = location.key
                continue

            if class_colors[class_name] != color:
                _warn(
                    f"[Audit] {schema_name}/{class_name}: color mismatch. "
                    f"Keeping {class_colors[class_name]} from {class_sources[class_name]}, "
                    f"ignoring {color} from {location.key}."
                )

    global_schemas = {
        schema_name: _build_global_schema(schema_name, class_colors, unlabeled_policy)
        for schema_name, class_colors in sorted(class_colors_by_schema.items())
    }

    for schema_name, schema in global_schemas.items():
        logging.info(
            "[Audit] Global schema %r has %d class(es): %s",
            schema_name,
            len(schema.semantic_classes),
            [sem_cls.name for sem_cls in schema.semantic_classes],
        )

    return global_schemas, has_errors


def _unify_documents(
    locations: Iterable[FieldLocation],
    global_schemas: dict[str, SemanticSchema],
    dry_run: bool,
) -> tuple[int, int, int]:
    doc_cache: dict[Path, Document] = {}
    dirty_docs: set[Path] = set()
    checked_fields = 0
    remapped_fields = 0
    skipped_fields = 0

    for location in locations:
        checked_fields += 1
        target_schema = global_schemas[location.schema_name]
        if _same_signature_ignore_schema_name(location.schema, target_schema):
            logging.info("[Pass2] %s already uses the global signature.", location.key)
            continue

        doc = doc_cache.get(location.doc_path)
        if doc is None:
            doc = Document.load_hdf5(location.doc_path)
            doc_cache[location.doc_path] = doc

        pcd_obj = doc.scene_items.get(location.pointcloud_id)
        if not isinstance(pcd_obj, PointCloudData):
            _error(f"[Pass2] {location.key}: point cloud not found after loading document.")
            skipped_fields += 1
            continue
        pcd = cast(PointCloudData, pcd_obj)

        fields = pcd.get_fields(location.field_name)
        if not fields or not isinstance(fields[0], SemanticSegmentation):
            _error(f"[Pass2] {location.key}: semantic field not found after loading document.")
            skipped_fields += 1
            continue

        field = cast(SemanticSegmentation, fields[0])
        unknown_ids = _ids_not_in_schema(field)
        if unknown_ids:
            _error(
                f"[Pass2] {location.key}: label id(s) not present in its source schema: "
                f"{unknown_ids}. Skipping to avoid changing unknown semantics."
            )
            skipped_fields += 1
            continue

        target_class_names = set(_classes_by_name(target_schema))
        unmapped_source_classes = [
            sem_cls.name
            for sem_cls in field.schema.semantic_classes
            if sem_cls.name not in target_class_names
            and int(np.count_nonzero(field.data == int(sem_cls.id))) > 0
        ]
        if unmapped_source_classes:
            _error(
                f"[Pass2] {location.key}: class(es) {unmapped_source_classes} have labels in "
                "the field but are not present in the global schema. Skipping."
            )
            skipped_fields += 1
            continue

        remap_pairs = _remap_pairs_by_name(field.schema, target_schema)
        logging.info("[Pass2] %s remap: %s", location.key, remap_pairs)

        if not dry_run:
            field.remap(remap_pairs)
            field.schema = target_schema
            dirty_docs.add(location.doc_path)
        remapped_fields += 1

    if dry_run:
        logging.info("[Pass2] Dry run requested; no documents were saved.")
    else:
        for doc_path in sorted(dirty_docs):
            doc_cache[doc_path].save_hdf5(doc_path)
            logging.info("[Save] Updated %s", doc_path)

    return checked_fields, remapped_fields, skipped_fields


def unify_semantic_schemas(
    dataset_fp: str,
    schema_name: Optional[str] = None,
    dry_run: bool = False,
    log_path: Optional[str] = None,
    unlabeled_policy: UnlabeledPolicy = "include",
) -> None:
    _import_geon_types()

    dataset_path = Path(dataset_fp)
    if log_path is None:
        log_path = str(dataset_path / "unify_semantic_schemas.log")
    _configure_logging(Path(log_path))

    locations = _scan_locations(dataset_fp, schema_name)
    logging.info("[Pass1] Found %d semantic field(s).", len(locations))
    if schema_name is not None:
        logging.info("[Pass1] Filtering to schema name %r.", schema_name)
    logging.info("[Pass1] Unlabeled policy: %s.", unlabeled_policy)
    if not locations:
        logging.info("[Done] No semantic schemas matched.")
        return

    global_schemas, has_errors = _audit_schemas(locations, unlabeled_policy)
    if has_errors:
        _error("[Done] Audit failed. Fix duplicate class names before unifying schemas.")
        return

    checked_fields, remapped_fields, skipped_fields = _unify_documents(
        locations,
        global_schemas,
        dry_run=dry_run,
    )
    logging.info(
        "[Done] Checked %d field(s), remapped %d field(s), skipped %d field(s).",
        checked_fields,
        remapped_fields,
        skipped_fields,
    )


def parse_arguments():
    parser = ArgumentParser(
        description=(
            "Unify semantic schemas with the same schema name across a GEON dataset. "
            "Classes are matched by name and remapped to one alphabetically sorted global schema."
        )
    )
    parser.add_argument("-d", "--dataset", required=True, type=str, help="Dataset folder path.")
    parser.add_argument(
        "-s",
        "--schema-name",
        default=None,
        type=str,
        help="Optional schema name to unify. If omitted, all schema names are unified.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Audit and report planned remaps without saving documents.",
    )
    parser.add_argument(
        "--log",
        default=None,
        type=str,
        help="Optional log file path. Defaults to <dataset>/unify_semantic_schemas.log.",
    )
    parser.add_argument(
        "--unlabeled",
        choices=("include", "empty"),
        default="include",
        help=(
            'Global schema seed policy. "include" adds _unlabeled with id -1 '
            'using an existing color when found. "empty" starts from class names only '
            "and omits _unlabeled from the global schema."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    unify_semantic_schemas(
        args.dataset,
        schema_name=args.schema_name,
        dry_run=args.dry_run,
        log_path=args.log,
        unlabeled_policy=args.unlabeled,
    )
