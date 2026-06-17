import h5py
import json
import numpy as np
import pytest
from types import SimpleNamespace

from geon.data.document import Document
from geon.data.cellcomplex import (
    CellComplex,
    CellComplexData,
    EdgeCell,
    PointCloudRef,
    VertexCell,
)
from geon.data.pointcloud import FieldType, PointCloudData, SemanticClass, SemanticSchema
from geon.io.dataset import Dataset, DocumentReference, RefLoadedState, RefModState
from geon.rendering.cellcomplex import CellComplexLayer


def _schema(name: str = "cell_schema") -> SemanticSchema:
    return SemanticSchema(
        name=name,
        semantic_classes=[
            SemanticClass(-1, "_unlabeled", (204, 204, 204)),
            SemanticClass(1, "node", (10, 20, 30)),
            SemanticClass(2, "beam", (30, 40, 50)),
        ],
    )


def test_vertex_position_only_constructor() -> None:
    vertex = VertexCell(position=(1.0, 2.0, 3.0))

    assert vertex.position == (1.0, 2.0, 3.0)
    assert vertex.boundary == []
    assert vertex.semantic_attributes == {}
    assert vertex.attributes == {}
    assert vertex.geometry_refs == []
    assert isinstance(vertex.id, str)


def test_from_txt_single_and_multi_row(tmp_path) -> None:
    single_path = tmp_path / "single.txt"
    single_path.write_text("1,2,3\n", encoding="utf-8")
    single = CellComplexData.from_txt(str(single_path))
    assert len(single.vertices) == 1
    assert single.vertices[0].position == (1.0, 2.0, 3.0)

    multi_path = tmp_path / "multi.txt"
    multi_path.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
    multi = CellComplexData.from_txt(str(multi_path))
    assert [v.position for v in multi.vertices] == [
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    ]


def test_edge_validation_uses_vertex_ids() -> None:
    a = VertexCell(position=(0.0, 0.0, 0.0), id="a")
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    complex_data = CellComplexData(vertices=[a, b])

    edge = complex_data.build_edge("a", "b")

    assert edge.boundary == ["a", "b"]
    edge.validate(complex_data)


def test_point_cloud_ref_properties_use_ref_id() -> None:
    ref = PointCloudRef(ref_id="pcd-1", field_name="instances", instance_id=7)

    assert ref.get_properties() == {
        "ID": "pcd-1",
        "field_name": "instances",
        "instance_id": 7,
    }


def test_layer_selection_stores_cell_ids_by_dimension() -> None:
    a = VertexCell(position=(0.0, 0.0, 0.0), id="a")
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    complex_data = CellComplexData(vertices=[a, b])
    edge = complex_data.build_edge("a", "b")
    layer = CellComplexLayer(complex_data)

    layer.active_selection = {"a", edge.id, "missing"}

    assert layer.active_selection == {"a", edge.id}
    assert layer.selected_ids_by_dim(0) == ["a"]
    assert layer.selected_ids_by_dim(1) == [edge.id]


def test_layer_selection_modes_replace_add_remove() -> None:
    a = VertexCell(position=(0.0, 0.0, 0.0), id="a")
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    layer = CellComplexLayer(CellComplexData(vertices=[a, b]))
    layer.active_selection = {"a"}

    assert layer._selection_with_pick(SimpleNamespace(ctrl=False, shift=False), "b") == {"b"}
    assert layer._selection_with_pick(SimpleNamespace(ctrl=False, shift=True), "b") == {"a", "b"}
    assert layer._selection_with_pick(SimpleNamespace(ctrl=True, shift=False), "a") == set()


def test_delete_vertices_removes_incident_edges() -> None:
    a = VertexCell(position=(0.0, 0.0, 0.0), id="a")
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    c = VertexCell(position=(2.0, 0.0, 0.0), id="c")
    complex_data = CellComplexData(vertices=[a, b, c])
    ab = complex_data.build_edge("a", "b")
    bc = complex_data.build_edge("b", "c")

    removed_vertices, removed_edges = complex_data.remove_vertices({"b"})

    assert [vertex.id for vertex in removed_vertices] == ["b"]
    assert {edge.id for edge in removed_edges} == {ab.id, bc.id}
    assert [vertex.id for vertex in complex_data.vertices] == ["a", "c"]
    assert complex_data.edges == []


def test_set_vertex_positions_moves_nodes_and_preserves_edges() -> None:
    a = VertexCell(position=(0.0, 0.0, 0.0), id="a")
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    complex_data = CellComplexData(vertices=[a, b])
    edge = complex_data.build_edge("a", "b")

    complex_data.set_vertex_positions({"a": (2.0, 3.0, 4.0), "missing": (9.0, 9.0, 9.0)})

    assert complex_data.get_vertex_by_id("a").position == (2.0, 3.0, 4.0)
    assert complex_data.get_vertex_by_id("b").position == (1.0, 0.0, 0.0)
    assert complex_data.edges[0].id == edge.id
    assert complex_data.edges[0].boundary == ["a", "b"]


def test_hdf5_roundtrip(tmp_path) -> None:
    schema = _schema()
    a = VertexCell(
        position=(0.0, 0.0, 0.0),
        id="a",
        semantic_attributes={"role": 1},
        attributes={"height": 2.5},
    )
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    complex_data = CellComplexData(
        vertices=[a, b],
        semantic_attribute_schemas={0: {"role": schema}},
    )
    complex_data.build_edge("a", "b")

    path = tmp_path / "cellcomplex.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("cellcomplex")
        complex_data.save_hdf5(group)

    with h5py.File(path, "r") as h5:
        loaded = CellComplexData.load_hdf5(h5["cellcomplex"])

    assert isinstance(loaded, CellComplex)
    assert len(loaded.vertices) == 2
    assert len(loaded.edges) == 1
    assert loaded.vertices[0].id == "a"
    assert "role" in loaded.vertices[0].semantic_attributes
    assert loaded.vertices[0].semantic_attributes["role"] == 1
    assert loaded.vertices[1].semantic_attributes["role"] == -1
    assert loaded.semantic_attribute_schemas[0]["role"].name == schema.name
    assert loaded.semantic_attribute_schemas[0]["role"].signature() == schema.signature()
    assert loaded.vertices[0].attributes == {"height": 2.5}
    assert loaded.vertices[1].attributes == {"height": None}
    assert loaded.edges[0].boundary == ["a", "b"]

    with h5py.File(path, "r") as h5:
        vertex_group = h5["cellcomplex"]["vertices"]["a"]
        assert "semantic_attributes" in vertex_group.attrs
        assert "semantic_type" not in vertex_group.attrs
        assert "semantic_attribute_schemas" in h5["cellcomplex"]


def test_hdf5_load_migrates_legacy_per_cell_semantic_schema(tmp_path) -> None:
    schema = _schema("legacy_schema")
    path = tmp_path / "legacy_cellcomplex.h5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("cellcomplex")
        group.attrs["type_id"] = CellComplexData.get_type_id()
        group.attrs["id"] = "CC_legacy"
        vertices_group = group.create_group("vertices")
        vertex_group = vertices_group.create_group("a")
        vertex_group.attrs["id"] = "a"
        vertex_group.attrs["dim"] = 0
        vertex_group.attrs["semantic_attributes"] = json.dumps({
            "role": {
                "value": {"id": 1, "name": "node", "color": [10, 20, 30]},
                "schema_name": schema.name,
                "schema": schema.to_dict(),
            }
        })
        vertex_group.attrs["attributes"] = "{}"
        vertex_group.attrs["geometry_refs"] = "[]"
        dt = h5py.string_dtype(encoding="utf-8")
        vertex_group.create_dataset("boundary", data=np.asarray([], dtype=dt), dtype=dt)
        vertex_group.create_dataset("position", data=np.asarray([0.0, 0.0, 0.0], dtype=np.float64))
        group.create_group("edges")

    with h5py.File(path, "r") as h5:
        loaded = CellComplexData.load_hdf5(h5["cellcomplex"])

    assert loaded.vertices[0].semantic_attributes["role"] == 1
    assert loaded.semantic_attribute_schemas[0]["role"].name == schema.name
    assert loaded.semantic_attribute_schemas[0]["role"].signature() == schema.signature()


def test_unify_semantic_attributes_per_dimension() -> None:
    schema = _schema()
    a = VertexCell(
        position=(0.0, 0.0, 0.0),
        id="a",
        semantic_attributes={"role": 1},
    )
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")

    complex_data = CellComplexData(
        vertices=[a, b],
        semantic_attribute_schemas={0: {"role": schema}},
    )

    assert complex_data.vertices[1].semantic_attributes["role"] == -1


def test_unregistered_semantic_attribute_raises() -> None:
    a = VertexCell(
        position=(0.0, 0.0, 0.0),
        id="a",
        semantic_attributes={"role": 1},
    )

    with pytest.raises(ValueError, match="no schema registered"):
        CellComplexData(vertices=[a])


def test_semantic_attributes_are_independent_per_dimension() -> None:
    vertex_schema = _schema("vertex_schema")
    edge_schema = _schema("edge_schema")
    a = VertexCell(
        position=(0.0, 0.0, 0.0),
        id="a",
        semantic_attributes={"role": 1},
    )
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    edge = EdgeCell(
        id="e",
        boundary=["a", "b"],
        semantic_attributes={"role": 2},
    )

    complex_data = CellComplexData(
        vertices=[a, b],
        edges=[edge],
        semantic_attribute_schemas={
            0: {"role": vertex_schema},
            1: {"role": edge_schema},
        },
    )

    assert complex_data.semantic_attribute_schemas[0]["role"].name == "vertex_schema"
    assert complex_data.semantic_attribute_schemas[1]["role"].name == "edge_schema"


def test_cellcomplex_layer_colors_by_active_semantic_attribute_per_dimension() -> None:
    vertex_schema = _schema("vertex_schema")
    edge_schema = _schema("edge_schema")
    a = VertexCell(
        position=(0.0, 0.0, 0.0),
        id="a",
        semantic_attributes={"role": 1},
    )
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    edge = EdgeCell(
        id="e",
        boundary=["a", "b"],
        semantic_attributes={"kind": 2},
    )
    layer = CellComplexLayer(
        CellComplexData(
            vertices=[a, b],
            edges=[edge],
            semantic_attribute_schemas={
                0: {"role": vertex_schema},
                1: {"kind": edge_schema},
            },
        )
    )

    assert layer._base_color(layer.data.vertices[0]) == layer.default_color
    layer.set_active_semantic_attribute(0, "role")
    layer.set_active_semantic_attribute(1, "kind")

    assert layer._base_color(layer.data.vertices[0]) == (10, 20, 30)
    assert layer._edge_color(layer.data.edges[0]) == (30, 40, 50)


def test_cellcomplex_layer_cleans_missing_active_semantic_attribute() -> None:
    schema = _schema()
    cell = VertexCell(position=(0.0, 0.0, 0.0), id="a", semantic_attributes={"role": 1})
    layer = CellComplexLayer(
        CellComplexData(vertices=[cell], semantic_attribute_schemas={0: {"role": schema}})
    )
    layer.set_active_semantic_attribute(0, "role")

    layer.data.semantic_attribute_schemas[0].clear()
    layer.update()

    assert layer.active_semantic_attribute_by_dim[0] is None


def test_cellcomplex_layer_gizmo_origin_uses_selected_vertices_only() -> None:
    a = VertexCell(position=(0.0, 0.0, 0.0), id="a")
    b = VertexCell(position=(2.0, 0.0, 0.0), id="b")
    edge = EdgeCell(id="e", boundary=["a", "b"])
    layer = CellComplexLayer(CellComplexData(vertices=[a, b], edges=[edge]))

    layer.active_selection = {"a", "b", "e"}
    layer.set_node_gizmo_enabled(True)
    layer.update()

    assert layer.selected_vertex_centroid() == (1.0, 0.0, 0.0)
    assert layer.node_gizmo_origin == (1.0, 0.0, 0.0)


def test_cellcomplex_layer_gizmo_hides_without_selected_vertices() -> None:
    a = VertexCell(position=(0.0, 0.0, 0.0), id="a")
    b = VertexCell(position=(2.0, 0.0, 0.0), id="b")
    edge = EdgeCell(id="e", boundary=["a", "b"])
    layer = CellComplexLayer(CellComplexData(vertices=[a, b], edges=[edge]))

    layer.active_selection = {"e"}
    layer.set_node_gizmo_enabled(True)
    layer.update()

    assert layer.selected_vertex_centroid() is None
    assert layer.node_gizmo_origin is None


def test_remap_semantic_attribute_updates_values_and_schema() -> None:
    old_schema = _schema("shared")
    new_schema = SemanticSchema(
        name="shared",
        semantic_classes=[
            SemanticClass(-1, "_unlabeled", (1, 1, 1)),
            SemanticClass(10, "node_new", (100, 110, 120)),
        ],
    )
    a = VertexCell(
        position=(0.0, 0.0, 0.0),
        id="a",
        semantic_attributes={"role": 1},
    )
    b = VertexCell(
        position=(1.0, 0.0, 0.0),
        id="b",
        semantic_attributes={"role": 2},
    )
    complex_data = CellComplexData(
        vertices=[a, b],
        semantic_attribute_schemas={0: {"role": old_schema}},
    )

    complex_data.remap_semantic_attribute(0, "role", [(1, 10)], new_schema)

    assert complex_data.vertices[0].semantic_attributes["role"] == 10
    assert complex_data.vertices[1].semantic_attributes["role"] == -1
    assert complex_data.semantic_attribute_schemas[0]["role"] is new_schema


def test_matching_semantic_attribute_schemas_are_reported_once_per_attribute() -> None:
    schema = _schema("shared")
    complex_data = CellComplexData(semantic_attribute_schemas={0: {"role": schema}})

    matches = complex_data.get_matching_semantic_attribute_schemas(schema)

    assert list(matches.keys()) == ["0/role/shared"]
    assert matches["0/role/shared"].signature() == schema.signature()


def test_add_semantic_attribute_fills_dimension_and_activates_default_value() -> None:
    schema = _schema("shared")
    a = VertexCell(position=(0.0, 0.0, 0.0), id="a")
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    edge = EdgeCell(id="e", boundary=["a", "b"])
    complex_data = CellComplexData(vertices=[a, b], edges=[edge])

    complex_data.add_semantic_attribute(0, "role", schema)

    assert [cell.semantic_attributes["role"] for cell in complex_data.vertices] == [-1, -1]
    assert "role" not in complex_data.edges[0].semantic_attributes
    assert complex_data.semantic_attribute_schemas[0]["role"] is schema


def test_add_semantic_attribute_rejects_spaces_and_duplicates() -> None:
    schema = _schema("shared")
    complex_data = CellComplexData(vertices=[VertexCell(position=(0.0, 0.0, 0.0), id="a")])

    with pytest.raises(ValueError, match="spaces"):
        complex_data.add_semantic_attribute(0, "bad name", schema)

    complex_data.add_semantic_attribute(0, "role", schema)
    with pytest.raises(ValueError, match="already exists"):
        complex_data.add_semantic_attribute(0, "role", schema)


def test_add_semantic_attribute_allows_empty_dimension() -> None:
    schema = _schema("shared")
    complex_data = CellComplexData(vertices=[VertexCell(position=(0.0, 0.0, 0.0), id="a")])

    complex_data.add_semantic_attribute(1, "semantic", schema)
    edge = complex_data.build_edge("a", "a")

    assert complex_data.semantic_attribute_schemas[1]["semantic"] is schema
    assert edge.semantic_attributes["semantic"] == -1


def test_delete_semantic_attribute_removes_schema_and_cell_values() -> None:
    schema = _schema("shared")
    complex_data = CellComplexData(vertices=[
        VertexCell(position=(0.0, 0.0, 0.0), id="a"),
        VertexCell(position=(1.0, 0.0, 0.0), id="b"),
    ])
    complex_data.add_semantic_attribute(0, "role", schema)
    complex_data.vertices[0].semantic_attributes["role"] = 1

    complex_data.delete_semantic_attribute(0, "role")

    assert "role" not in complex_data.semantic_attribute_schemas[0]
    assert all("role" not in cell.semantic_attributes for cell in complex_data.vertices)


def test_dataset_global_schema_update_remaps_pointcloud_and_cellcomplex(tmp_path) -> None:
    old_schema = _schema("shared")
    new_schema = SemanticSchema(
        name="shared",
        semantic_classes=[
            SemanticClass(-1, "_unlabeled", (1, 1, 1)),
            SemanticClass(10, "node_new", (100, 110, 120)),
            SemanticClass(20, "beam_new", (120, 130, 140)),
        ],
    )

    doc = Document("doc")
    pcd = PointCloudData(np.zeros((2, 3), dtype=np.float32))
    pcd.add_field(
        name="sem",
        data=np.asarray([[1], [2]], dtype=np.int32),
        field_type=FieldType.SEMANTIC,
        schema=old_schema,
    )
    cell_complex = CellComplexData(vertices=[
        VertexCell(
            position=(0.0, 0.0, 0.0),
            id="a",
            semantic_attributes={"role": 1},
        )
    ], semantic_attribute_schemas={0: {"role": old_schema}})
    doc.add_data(pcd)
    doc.add_data(cell_complex)
    path = tmp_path / "doc.h5"
    doc.save_hdf5(path)

    dataset = Dataset()
    ref = DocumentReference("doc", str(path), RefModState.SAVED, RefLoadedState.ACTIVE)
    dataset._doc_refs.append(ref)
    dataset._loaded_docs["doc"] = doc

    matches = dataset.get_matching_schemas(old_schema)
    assert {key.split("/")[0] for key in matches} == {"point", "cell"}

    dataset.update_semantic_schema(old_schema, new_schema, [(1, 10), (2, 20)])

    sem_field = pcd.get_fields("sem")[0]
    assert sem_field.data.reshape(-1).tolist() == [10, 20]
    assert sem_field.schema is new_schema
    assert cell_complex.vertices[0].semantic_attributes["role"] == 10
    assert cell_complex.semantic_attribute_schemas[0]["role"] is new_schema
