import h5py
from types import SimpleNamespace

from geon.data.cellcomplex import CellComplex, CellComplexData, PointCloudRef, VertexCell
from geon.data.pointcloud import SemanticClass
from geon.rendering.cellcomplex import CellComplexLayer


def test_vertex_position_only_constructor() -> None:
    vertex = VertexCell(position=(1.0, 2.0, 3.0))

    assert vertex.position == (1.0, 2.0, 3.0)
    assert vertex.boundary == []
    assert vertex.semantic_type is None
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


def test_hdf5_roundtrip(tmp_path) -> None:
    sem = SemanticClass(1, "node", (10, 20, 30))
    a = VertexCell(
        position=(0.0, 0.0, 0.0),
        id="a",
        semantic_type=sem,
        attributes={"height": 2.5},
    )
    b = VertexCell(position=(1.0, 0.0, 0.0), id="b")
    complex_data = CellComplexData(vertices=[a, b])
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
    assert loaded.vertices[0].semantic_type == sem
    assert loaded.vertices[0].attributes == {"height": 2.5}
    assert loaded.edges[0].boundary == ["a", "b"]
