import json
import os
from copy import deepcopy
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ifcopenshell = pytest.importorskip("ifcopenshell")
from geon.io.ifc import (default_mapping, export_ifc, prepare_objects, available_ifc_classes,
                          predefined_types, save_mapping, load_mapping)
from geon.data.mesh import MeshData, MeshLevel
from geon.data.pointcloud import FieldType, SemanticSchema, SemanticClass


def bridge_mesh():
    mapping=default_mapping()
    vertices=np.array([[x,y,z] for i in range(21) for x,y,z in [(i*2,0,0),(i*2+1,0,0),(i*2,1,0)]])
    mesh=MeshData(vertices,np.arange(63).reshape(-1,3),"Bridge",face_object_ids=np.arange(21))
    classes=[SemanticClass(int(key),value['name'],(100,150,200)) for key,value in mapping['classes'].items()]
    mesh.add_field('class',np.arange(21),FieldType.SEMANTIC,schema=SemanticSchema('BRIDGE_',classes))
    mesh.add_field('patches',np.arange(21),FieldType.INSTANCE)
    return mesh,mapping


def test_all_bridge_classes_export_with_valid_schema_and_face_groups(tmp_path):
    mesh,mapping=bridge_mesh()
    mapping['face_field']={'level':'face','name':'patches'}
    path=tmp_path/'bridge.ifc'
    exported=export_ifc(mesh,path,mapping)
    assert path.exists()
    model=ifcopenshell.open(str(path))
    assert model.schema_identifier=='IFC4X3_ADD2'
    assert len(model.by_type('IfcShapeAspect'))==21
    assert len(model.by_type('IfcPolygonalFaceSet'))==21
    assert all(item.Closed is False for item in model.by_type('IfcPolygonalFaceSet'))
    assert len(model.by_type('IfcElement'))==21
    assert model.by_type('IfcElementAssembly')[0].PredefinedType=='ABUTMENT'
    assert model.by_type('IfcCourse')[0].PredefinedType=='BALLASTBED'
    assert len(model.by_type('IfcRelContainedInSpatialStructure')[0].RelatedElements)==20
    assert model.by_type('IfcRelAdheresToElement')[0].RelatingElement.is_a('IfcPavement') or model.by_type('IfcRelAdheresToElement')[0].RelatingElement.is_a('IfcSlab')
    from ifcopenshell import validate
    logger=validate.json_logger()
    validate.validate(model,logger)
    assert not logger.statements


def test_object_level_semantics_and_vertex_majority(tmp_path):
    mesh,mapping=bridge_mesh()
    object_semantic=mesh.add_field('object_class',np.arange(21),FieldType.SEMANTIC,
        level='object',schema=mesh.get_fields(names='class')[0].schema)
    mapping['semantic_field']={'level':'object','name':'object_class'}
    mapping['object_field']=None
    _,_,objects=prepare_objects(mesh,mapping)
    assert len(objects)==21
    mapping['classes']['0']={'name':'abutment','ifc_class':'IfcBridgePart','predefined_type':'ABUTMENT'}
    model=export_ifc(mesh,tmp_path/'spatial.ifc',mapping)
    part=model.by_type('IfcBridgePart')[0]
    assert part.UsageType=='NOTDEFINED'
    mesh.add_field('vertex_class',np.repeat(np.arange(21),3),FieldType.SEMANTIC,
        level='vertex',schema=object_semantic.schema)
    mapping['semantic_field']={'level':'vertex','name':'vertex_class'}
    with pytest.raises(ValueError,match='object separation'):
        prepare_objects(mesh,mapping)
    mapping['object_field']={'level':'face','name':'patches'}
    _,_,objects=prepare_objects(mesh,mapping)
    assert [item[1] for item in objects]==list(range(21))


def test_mapping_json_validation_and_ifc_dialog(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from geon.ui.ifc_export_dialog import IfcExportDialog
    app=QApplication.instance() or QApplication([])
    mesh,mapping=bridge_mesh()
    path=tmp_path/'ifc_map.json'
    save_mapping(mapping,path)
    assert load_mapping(path)==mapping
    dialog=IfcExportDialog(mesh)
    assert dialog.table.rowCount()==21
    assert dialog.object_combo.currentData()=={'level':'face','name':'patches'}
    assert dialog.current_mapping()['classes']['12']['ifc_class']=='IfcCourse'
    assert dialog.current_mapping()['classes']['13']['predefined_type']=='SLEEPER'
    assert 'IfcBearing' in available_ifc_classes()
    assert 'ABUTMENT' in predefined_types('IfcElementAssembly')
    dialog.accept()
    assert dialog.mapping['schema']=='IFC4X3_ADD2'
    dialog.close()
    mapping['classes']['0']['ifc_class']='IfcNotARealClass'
    with pytest.raises(ValueError,match='valid IFC4.3'):
        prepare_objects(mesh,mapping)
    path.write_text(json.dumps({'version':1,'schema':'IFC2X3','classes':{}}))
    with pytest.raises(ValueError,match='IFC4X3_ADD2'):
        load_mapping(path)


@pytest.mark.parametrize('group_faces', [False, True])
def test_export_unequal_face_groups_and_mixed_polygons(tmp_path, group_faces):
    mapping = default_mapping()
    vertices, faces = [], []
    for index in range(11):
        count = 3 + index % 3
        angles = np.arange(count) * (2*np.pi/count)
        offset = len(vertices)
        vertices.extend(np.column_stack((index*3 + np.cos(angles), np.sin(angles), np.zeros(count))))
        faces.append(list(range(offset, offset+count)))
    mesh = MeshData(vertices, faces, 'Mixed face groups')
    mesh.add_field('class', np.zeros(11, dtype=np.int32), FieldType.SEMANTIC,
                   schema=SemanticSchema('BRIDGE_', [SemanticClass(0, 'abutment', (150, 62, 63))]))
    mesh.add_field('patches', np.full(11, 7, dtype=np.int32), FieldType.INSTANCE)
    mesh.add_field('face_groups', np.arange(11, dtype=np.int32), FieldType.INSTANCE)
    if group_faces:
        mapping['face_field'] = {'level': 'face', 'name': 'face_groups'}
    target = tmp_path/'mixed.ifc'
    export_ifc(mesh, target, mapping)
    model = ifcopenshell.open(str(target))
    product = model.by_type('IfcElementAssembly')[0]
    assert len(product.Representation.Representations) == 1
    representation = product.Representation.Representations[0]
    assert len(representation.Items) == (11 if group_faces else 1)
    polygons = [face for item in representation.Items for face in item.Faces]
    assert sorted(len(face.CoordIndex) for face in polygons) == sorted(map(len, faces))
    aspects = model.by_type('IfcShapeAspect')
    assert len(aspects) == (11 if group_faces else 0)
    if group_faces:
        by_name = {aspect.Name: aspect for aspect in aspects}
        for index, item in enumerate(representation.Items):
            aspect = by_name[f'face_groups={index}']
            assert aspect.ShapeRepresentations[0].Items == (item,)
            assert aspect.PartOfProductDefinitionShape == product.Representation
    from ifcopenshell import validate
    logger = validate.json_logger()
    validate.validate(model, logger)
    assert not logger.statements
