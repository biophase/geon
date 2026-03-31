from pathlib import Path

from geon.data.pointcloud import SemanticClass, SemanticSchema
from geon.io.dataset import Dataset


def _schema(name: str, classes: list[tuple[int, str, tuple[int, int, int]]]) -> SemanticSchema:
    return SemanticSchema(
        name=name,
        semantic_classes=[SemanticClass(class_id, class_name, color) for class_id, class_name, color in classes],
    )


def test_semantic_schema_json_round_trip(tmp_path: Path):
    schema = _schema(
        "source",
        [
            (-1, "_unlabeled", (128, 128, 128)),
            (0, "ground", (10, 20, 30)),
            (1, "roof", (40, 50, 60)),
        ],
    )
    json_path = tmp_path / "schema.json"

    schema.to_json(str(json_path))
    loaded = SemanticSchema.from_json(str(json_path))

    assert loaded.to_dict() == schema.to_dict()


def test_align_to_schema_uses_target_ids_and_colors_and_appends_missing():
    active = _schema(
        "active",
        [
            (-1, "_unlabeled", (1, 1, 1)),
            (0, "ground", (10, 20, 30)),
            (1, "roof", (40, 50, 60)),
            (2, "tree", (70, 80, 90)),
        ],
    )
    target = _schema(
        "target",
        [
            (-1, "_unlabeled", (9, 9, 9)),
            (0, "roof", (101, 102, 103)),
            (1, "ground", (111, 112, 113)),
        ],
    )

    result = active.align_to_schema(target)

    assert result.partial_success is True
    assert result.missing_in_target_names == ["tree"]
    assert result.old_to_new_ids == [(-1, -1), (1, 0), (0, 1), (2, 2)]
    assert result.aligned_schema.by_id(-1).color == (9, 9, 9)
    assert result.aligned_schema.by_id(0).name == "roof"
    assert result.aligned_schema.by_id(0).color == (101, 102, 103)
    assert result.aligned_schema.by_id(1).name == "ground"
    assert result.aligned_schema.by_id(1).color == (111, 112, 113)
    assert result.aligned_schema.by_id(2).name == "tree"
    assert result.aligned_schema.by_id(2).color == (70, 80, 90)


def test_align_to_schema_rejects_duplicate_names():
    active = _schema(
        "active",
        [
            (-1, "_unlabeled", (1, 1, 1)),
            (0, "ground", (10, 20, 30)),
            (1, "ground", (40, 50, 60)),
        ],
    )
    target = _schema("target", [(-1, "_unlabeled", (9, 9, 9))])

    try:
        active.align_to_schema(target)
    except ValueError as exc:
        assert "duplicate class name" in str(exc)
    else:
        raise AssertionError("Expected duplicate-name alignment to fail.")


def test_dataset_get_matching_schemas_uses_supplied_schema(monkeypatch):
    schema_a = _schema("shared", [(-1, "_unlabeled", (1, 1, 1)), (0, "ground", (2, 2, 2))])
    schema_b = _schema("different", [(-1, "_unlabeled", (1, 1, 1)), (0, "roof", (3, 3, 3))])
    dataset = Dataset()

    monkeypatch.setattr(
        dataset,
        "_get_semantic_schemas",
        lambda: (
            {"ref/doc/field/shared": schema_a},
            {"loaded/doc/field/different": schema_b},
        ),
    )

    matches = dataset.get_matching_schemas(schema_a)

    assert list(matches.keys()) == ["ref/doc/field/shared"]
    assert matches["ref/doc/field/shared"].signature() == schema_a.signature()
