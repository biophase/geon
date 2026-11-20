from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import os
import json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import h5py
import numpy as np
import pytest
import vtk
from vtk.util import numpy_support as ns
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtCore import QThreadPool

from geon.data.mesh import MeshData, MeshLevel
from geon.data.pointcloud import PointCloudData, FieldType, SemanticSchema, SemanticClass
from geon.data.document import Document
from geon.data.definitions import ColorMap
from geon.rendering.mesh import MeshLayer
from geon.rendering.scene import Scene
from geon.tools.command_manager import CommandManager
from geon.tools.controller import ToolController
from geon.tools.tool_context import ToolContext
from geon.tools.reconstruction import ReconstructMeshTool
from geon.algorithms.reconstruction.transfer import (default_config, reconstruct_cloud,
    assign_points_to_faces, point_triangle_distances, deviation_surface, corner_curves)
from geon.ui.context_ribbon import ContextRibbon
from geon.ui.color_channels import ColorChannels
from geon.ui.color_histogram import calculate_histogram
from geon.ui.reconstruction_dialog import ReconstructionDialog, GROUPS
from geon.io.color_maps import scan_maps, save_map
from geon.io.dataset import Dataset, DocumentReference


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def mesh():
    return MeshData([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], [[0,1,2], [0,2,3]],
                    face_object_ids=[10,20])


def schema():
    return SemanticSchema("test", [SemanticClass(0,"A",(255,0,0)), SemanticClass(1,"B",(0,255,0))])


def test_mesh_fields_all_levels_and_persistence(tmp_path):
    data = mesh()
    assert data.edges.shape == (5,2)
    for level in MeshLevel:
        count = data.element_count(level)
        data.add_field("labels", np.arange(count)%2, FieldType.SEMANTIC, level=level, schema=schema())
        data.add_field("ids", np.arange(count), FieldType.INSTANCE, level=level)
        scalar = data.add_field("distance", np.linspace(0,1,count), FieldType.SCALAR, level=level)
        scalar.color_map = ColorMap("map-"+level.value)
    data.metadata = {"diagnostics": ["large metadata"*10000]}
    doc = Document("mesh")
    doc.add_data(data)
    path = tmp_path/"mesh.h5"
    doc.save_hdf5(path)
    loaded = Document.load_hdf5(path).scene_items[data.id]
    np.testing.assert_array_equal(loaded.face_object_ids, [10,20])
    for level in MeshLevel:
        for old in data.get_fields(level=level):
            new = loaded.get_fields(names=old.name, level=level)[0]
            np.testing.assert_array_equal(old.data, new.data)
            assert old.field_type == new.field_type
            if hasattr(old, "schema"):
                assert new.schema.name == old.schema.name
    assert loaded.metadata == data.metadata
    assert len(SemanticSchema.scan_h5(path)) == 4
    with pytest.raises(ValueError, match="length"):
        data.add_field("bad", [1], FieldType.INSTANCE, level="vertex")
    with pytest.raises(ValueError, match="Duplicate"):
        data.add_field("ids", [1,2], FieldType.INSTANCE)


def test_field_projection_and_extraction():
    data = mesh()
    vertex = data.add_field("class", [0,1,1,1], FieldType.SEMANTIC, level="vertex", schema=schema())
    objects = data.add_field("objects", [9,7], FieldType.INSTANCE, level="object")
    np.testing.assert_array_equal(data.face_values(vertex).ravel(), [1,1])
    np.testing.assert_array_equal(data.face_values(objects).ravel(), [9,7])
    data.add_field("edges", np.arange(5), FieldType.INSTANCE, level="edge")
    extracted = data.extract_selection([1])
    assert len(extracted.vertices) == 3 and len(extracted.faces) == 1
    assert len(extracted.get_fields(names="edges")[0].data) == 3
    np.testing.assert_array_equal(extracted.get_fields(names="class")[0].data.ravel(), [0,1,1])
    np.testing.assert_array_equal(extracted.get_fields(names="objects")[0].data.ravel(), [7])


def test_mesh_layer_field_colors_and_level_selection(app):
    data = mesh()
    data.add_field("classes", [0,1], FieldType.SEMANTIC, schema=schema())
    data.add_field("edges", np.arange(5), FieldType.INSTANCE, level="edge")
    layer = MeshLayer(data)
    renderer = vtk.vtkRenderer()
    layer.attach(renderer)
    layer.set_active_field("face", "classes")
    np.testing.assert_array_equal(ns.vtk_to_numpy(layer._poly.GetCellData().GetScalars()), [[255,0,0],[0,255,0]])
    layer.active_selection = np.array([1])
    snapshot = layer.capture_selection()
    layer.set_active_field("edge", "edges")
    assert layer.active_selection is None
    assert layer._element_actor.GetVisibility()
    assert layer._element_poly.GetNumberOfLines() == 5
    layer.restore_selection(snapshot)
    np.testing.assert_array_equal(layer.active_selection, [1])
    layer.set_opacity(.3)
    assert layer._actor.GetProperty().GetOpacity() == .3
    layer.set_visible(False)
    assert not any(actor.GetVisibility() for actor in layer.actors)
    layer.detach()
    assert renderer.GetActors().GetNumberOfItems() == 0


def cloud():
    x,y=np.meshgrid(np.linspace(0,1,21),np.linspace(0,1,21))
    x,y=x.ravel(),y.ravel()
    data=PointCloudData(np.column_stack([x,y,.001*np.sin(x*20)]))
    data.add_field("patches",np.zeros(len(x),dtype=np.int32),FieldType.INSTANCE)
    data.add_field("classes",(x>.5).astype(np.int32),FieldType.SEMANTIC,schema=schema())
    data.add_field("objects",np.where(x>.5,200,100).astype(np.int32),FieldType.INSTANCE)
    return data


def test_reconstruction_final_face_votes_and_bounded_deviation():
    data = cloud()
    before = data.points.copy()
    result_mesh,result = reconstruct_cloud(data,"patches",["classes","objects"])
    assert len(result_mesh.faces) >= 2
    np.testing.assert_array_equal(result_mesh.get_fields(names="patches")[0].data, 0)
    labels = result_mesh.get_fields(names="classes")[0]
    assert set(labels.data.ravel()) == {0,1}
    assert labels.schema is not data.get_fields(names="classes")[0].schema
    assert set(result_mesh.get_fields(names="objects")[0].data.ravel()) == {100,200}
    assert len(result.point_deviations) == len(data.points)
    assert np.median(result.point_deviations) < .003
    assert np.max(result.point_deviations) < .06  # Simplification may cut a border corner.
    vertices,faces,values = deviation_surface(result,data.points,np.zeros(len(data.points)),spacing=.01,max_vertices=1000)
    assert len(vertices)<=1000 and len(values)==len(vertices)
    assert len(faces)>len(result_mesh.faces)
    assert set(corner_curves(result)) == {"analytic","coplanar","outer"}
    assert not hasattr(result_mesh,"point_deviations")
    np.testing.assert_array_equal(data.points,before)


def test_exact_triangle_distances_and_final_face_assignment():
    triangles=np.array([[[0,0,0],[1,0,0],[0,1,0]]]*3,dtype=float)
    values=point_triangle_distances(np.array([[.2,.2,.3],[2,0,0],[0,0,0]]),triangles)
    np.testing.assert_allclose(values,[.3,1,0])
    result=SimpleNamespace(vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
        faces=np.array([[0,1,2],[0,2,3]]),face_patch_ids=np.array([1,1]),
        patches={1:SimpleNamespace(project=lambda xyz:xyz[:,:2],cylinder=None)})
    assignment=assign_points_to_faces(result,np.array([[.8,.2,0],[.2,.8,0],[-.1,.8,0],[0,0,0]]),
        np.array([1,1,1,-1]))
    np.testing.assert_array_equal(assignment,[0,1,1,-1])


def test_dialog_defaults_and_arbitrary_mapped_fields(app):
    from geon.rendering.pointcloud import PointCloudLayer
    layer=PointCloudLayer(cloud())
    dialog=ReconstructionDialog(layer)
    assert len(dialog.mapping_combos)==2
    assert set(dialog.inputs)=={name for names in GROUPS.values() for name in names}
    dialog.add_mapping("objects")
    assert len(dialog.mapping_combos)==3
    dialog.accept()
    assert dialog.config.min_points==150
    assert dialog.config.simplify_tolerance==.12
    assert dialog.config.max_plane_rmse==.5
    assert dialog.config.subsample_distance is None
    assert dialog.config.cylinder_trim_mode=="extent"
    dialog.close()


def preview_setup(app, splines=None):
    from geon.rendering.pointcloud import PointCloudLayer
    window=QMainWindow()
    window.dataset_manager=SimpleNamespace(_dataset=None)
    renderer=vtk.vtkRenderer()
    viewer=QWidget(window)
    viewer._renderer=renderer
    viewer.rerender=Mock()
    window.viewer=viewer
    window.color_channels=ColorChannels(window)
    scene=Scene(renderer)
    source=PointCloudLayer(cloud())
    scene.doc.add_data(source.data)
    scene.add_layer(source)
    scene.active_layer_id=source.id
    controller=ToolController(ContextRibbon())
    ctx=ToolContext(scene,viewer,controller)
    controller.ctx=ctx
    tool=ReconstructMeshTool(CommandManager(),ctx)
    controller._active_tool=tool
    tool._in_use=True
    tool.source=source
    tool.previous_active=source.id
    tool.source_visible=True
    tool.mesh_name="Preview"
    widget=tool.create_context_widget(window)
    mesh_result,result=reconstruct_cloud(source.data,"patches",["classes","objects"])
    surface=deviation_surface(result,source.data.points,np.zeros(len(source.data.points)),max_vertices=1000)
    tool._ready((mesh_result,corner_curves(result),surface,result.point_deviations,splines or {}))
    return tool,ctx,window,widget


def test_deviation_preview_exceeds_old_vertex_cap():
    result=SimpleNamespace(vertices=np.array([[0.,0,0],[1.,0,0],[0.,1,0]]),
        faces=np.tile([0,1,2],(26667,1)),face_patch_ids=np.zeros(26667,dtype=int),
        deviation_point_indices=np.array([0]),point_deviations=np.array([.02]))
    progress=Mock()
    vertices,faces,values=deviation_surface(result,np.array([[0.,0,0]]),np.array([0]),
                                           spacing=2,progress=progress)
    assert len(vertices)>80000 and len(faces)==26667
    np.testing.assert_allclose(values,.02)
    progress.assert_called()


def test_preview_histogram_uses_source_points_and_cancel_cleans_up(app):
    tool,ctx,window,widget=preview_setup(app)
    preview=tool.preview_layer
    assert preview.id not in ctx.scene.doc.scene_items
    tool.field_combo.setCurrentIndex(tool.field_combo.findData("__deviation__"))
    panel=window.color_channels
    assert panel.source_field is tool._deviation_field
    assert panel.axis_actions["bottom","numbers"].isChecked()
    edges,counts=calculate_histogram([panel.source_field.data.ravel()])
    values=panel.source_field.data.ravel()
    assert counts[0].sum()==np.count_nonzero((values>=edges[0]) & (values<=edges[-1]))
    assert counts[0].sum()>len(preview.data.faces)
    panel.show()
    panel.histogram_ready(panel.generation,edges,counts)
    tool.transparency.setValue(75)
    assert preview._actor.GetProperty().GetOpacity()==.25
    assert tool._deviation_actor.GetProperty().GetOpacity()==.25
    ctx.controller.deactivate_tool()
    assert preview.id not in ctx.scene.layers
    assert preview.id not in ctx.scene.doc.scene_items
    assert tool._surface is None and tool._deviation_field is None
    assert panel.source_field is None
    assert not panel.axis_actions["bottom","numbers"].isChecked()
    assert tool.source.visible
    assert ctx.scene.active_layer_id==tool.source.id
    QThreadPool.globalInstance().waitForDone()
    window.close()


def test_preview_accept_has_one_undoable_mesh_without_temporary_geometry(app):
    tool,ctx,window,widget=preview_setup(app)
    layer=tool.preview_layer
    tool.field_combo.setCurrentIndex(tool.field_combo.findData("__deviation__"))
    original_faces=len(layer.data.faces)
    tool._accept()
    assert layer.id in ctx.scene.doc.scene_items and layer.id in ctx.scene.layers
    assert ctx.scene.active_layer_id==layer.id
    assert layer.preview_field is None and layer.preview_map_changed is None
    assert len(layer.data.faces)==original_faces
    assert not any("deviation" in field.name.lower() for field in layer.data.get_fields())
    manager=ctx.controller.command_manager
    assert len(manager.undo_stack)==1
    manager.undo()
    assert layer.id not in ctx.scene.doc.scene_items and layer.id not in ctx.scene.layers
    manager.redo()
    assert layer.id in ctx.scene.doc.scene_items and layer.id in ctx.scene.layers
    ctx.scene.clear(delete_data=False)
    QThreadPool.globalInstance().waitForDone()
    window.close()


def test_mesh_color_maps_index_and_save_each_level(tmp_path, monkeypatch):
    data=mesh()
    for level in MeshLevel:
        field=data.add_field('value',np.arange(data.element_count(level),dtype=float),FieldType.SCALAR,level=level)
        field.color_map=ColorMap('level-'+level.value)
    doc=Document('mesh-maps')
    doc.add_data(data)
    path=tmp_path/'maps.h5'
    doc.save_hdf5(path)
    dataset=Dataset()
    dataset._doc_refs.append(DocumentReference(doc.name,str(path)))
    assert len(scan_maps(dataset))==4
    dataset._loaded_docs[doc.name]=doc
    field=data.get_fields(names='value',level='face')[0]
    edited=deepcopy(field.color_map)
    edited.max=42
    save_map(dataset,deepcopy(field.color_map),edited,data,field)
    with h5py.File(path) as handle:
        assert handle[f'document/{data.id}/fields/face/value/color_map'].attrs['max']==42
        assert 'max' not in handle[f'document/{data.id}/fields/vertex/value/color_map'].attrs
    assert len(scan_maps(dataset))==4


def test_mesh_scalar_color_editor_and_histogram(app, monkeypatch):
    data=mesh()
    data.add_field('scalar',[0.,1.],FieldType.SCALAR)
    layer=MeshLayer(data)
    renderer=vtk.vtkRenderer()
    layer.attach(renderer)
    layer.set_active_field('face','scalar')
    window=QMainWindow()
    window.dataset_manager=SimpleNamespace(_dataset=None)
    window.viewer=SimpleNamespace(rerender=Mock())
    panel=ColorChannels(window)
    panel.activate(layer)
    assert panel.source_field is data.get_fields(names='scalar')[0]
    panel.draft=ColorMap('gray',color_definitions=np.array([[0.,0.,0.],[1.,1.,1.]]),min=0.,max=2.)
    panel.changed()
    np.testing.assert_array_equal(ns.vtk_to_numpy(layer._poly.GetCellData().GetScalars()),[[0,0,0],[127,127,127]])
    edges,counts=calculate_histogram([panel.field.data.ravel()])
    assert counts[0].sum()==2
    panel.revert()
    panel.activate(None)
    layer.detach()
    window.close()


def test_worker_runs_on_snapshot_and_delivers_preview_on_gui_thread(app, monkeypatch):
    import time
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import QThread
    tool,ctx,window,widget=preview_setup(app)
    ctx.controller.deactivate_tool()
    original_ready=ReconstructMeshTool._ready
    def ready(self,payload):
        assert QThread.currentThread()==app.thread()
        original_ready(self,payload)
    monkeypatch.setattr(ReconstructMeshTool,'_ready',ready)
    def accept_dialog(self):
        self.accept()
        return self.result()
    monkeypatch.setattr(ReconstructionDialog,'exec',accept_dialog)
    ctx.controller.activate_tool('ReconstructMeshTool')
    running=ctx.controller.active_tool
    deadline=time.monotonic()+10
    while running.preview_layer is None and time.monotonic()<deadline:
        QTest.qWait(10)
    assert running.preview_layer is not None
    ctx.controller.cancel_background_tasks()
    app.processEvents()
    ctx.controller.deactivate_tool()
    assert len(ctx.scene.layers)==1
    assert len(ctx.scene.doc.scene_items)==1
    window.close()


def test_all_reconstruction_config_controls_are_exposed(app):
    from dataclasses import fields
    from geon.algorithms.reconstruction.core import Config
    from geon.rendering.pointcloud import PointCloudLayer
    dialog=ReconstructionDialog(PointCloudLayer(cloud()))
    assert set(dialog.inputs)=={field.name for field in fields(Config)}
    dialog.inputs['clean_first'].setChecked(True)
    assert not dialog.inputs['clean_border_pass'].isEnabled()
    assert dialog.inputs['stitch_mesh_boundaries'].isEnabled()
    dialog.inputs['clean_first'].setChecked(False)
    assert dialog.inputs['clean_border_pass'].isEnabled()
    dialog.close()


def test_reconstruction_log_exposes_fit_rejections_and_updates(app):
    from PyQt6.QtWidgets import QPlainTextEdit
    tool,ctx,window,widget=preview_setup(app)
    tool._set_status('Patch 25: rejected spline: insufficient support')
    tool.log_button.click()
    text=tool._log_dialog.findChild(QPlainTextEdit)
    assert 'Patch 25: rejected spline' in text.toPlainText()
    assert text.isReadOnly()
    tool._set_status('Patch 25: PCA height axis 1: accepted spline')
    assert 'PCA height axis 1' in text.toPlainText()
    assert all(entry.startswith('[elapsed ') and '; delta ' in entry for entry in tool._log_messages)
    tool._log_dialog.close()
    ctx.controller.deactivate_tool()
    window.close()


def test_reconstruction_ribbon_has_two_rows_and_bounded_status(app):
    tool,ctx,window,widget=preview_setup(app)
    grid=widget.layout()
    assert grid.rowCount()==2
    for index in range(grid.count()):
        row,column,row_span,column_span=grid.getItemPosition(index)
        assert row+row_span<=2
        assert grid.itemAt(index).widget().height()==18
    message='Pair 123/456: computing and validating surface intersections '*8
    tool._set_status(message)
    assert message in tool.status_label.toolTip()
    assert tool.status_label.width()==260
    assert tool.status_label.fontMetrics().horizontalAdvance(tool.status_label.text())<=260
    ctx.controller.deactivate_tool()
    window.close()


def test_reconstruction_failure_detail_is_in_timestamped_log(app,monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    tool,ctx,window,widget=preview_setup(app)
    dialogs=[]
    monkeypatch.setattr(QMessageBox,'exec',lambda self:dialogs.append(self.detailedText()))
    tool._failed('Patch 9: shared seam node is not on the accepted boundary')
    assert 'Reconstruction failed: Patch 9:' in dialogs[0]
    assert '[elapsed ' in dialogs[0]
    assert tool.preview_layer is None
    window.close()


@pytest.mark.parametrize('accept',[False,True])
def test_spline_overlays_are_optional_and_removed_after_preview(app,accept):
    grid=[np.array([[0.,0,0],[1.,0,0],[1.,1,0]])]
    overlay={25:dict(grid=grid,control_net=[np.array([[0.,0,.2],[.4,0,.2]])],
                     control_points=np.array([[0.,0,.2],[1.,0,.2]]))}
    tool,ctx,window,widget=preview_setup(app,overlay)
    assert tool.spline_grid_button.isEnabled()
    assert tool.spline_control_button.isEnabled()
    assert tool._spline_grid_actors[0].GetVisibility()
    assert not any(a.GetVisibility() for a in tool._spline_control_actors)
    tool.spline_control_button.setChecked(True)
    tool.corners_button.setChecked(False)
    assert all(a.GetVisibility() for a in tool._spline_control_actors+tool._spline_grid_actors)
    assert not any(a.GetVisibility() for a in tool._overlays)
    actors=tool._spline_control_actors+tool._spline_grid_actors
    if accept:
        tool._accept()
        assert len(ctx.scene.doc.scene_items)==2
    else:
        ctx.controller.deactivate_tool()
        assert len(ctx.scene.doc.scene_items)==1
    assert not tool._spline_control_actors and not tool._spline_grid_actors
    assert not any(ctx.viewer._renderer.HasViewProp(a) for a in actors)
    window.close()


def test_empty_reconstruction_error_includes_patch_rejection():
    from geon.algorithms.reconstruction.core import Config
    points=np.random.default_rng(74).normal(size=(300,3))
    data=PointCloudData(points)
    data.add_field('instances',np.full(len(points),25,dtype=np.int32),FieldType.INSTANCE)
    with pytest.raises(ValueError,match='Patch 25:'):
        reconstruct_cloud(data,'instances',config=Config(fit_cylinders=False,fit_bsplines=False))


@pytest.mark.parametrize('level',list(MeshLevel))
def test_mesh_classification_targets_level_and_survives_field_snapshot_restore(app,level):
    from geon.tools.mesh_fields import MeshAnnotateTool
    data=mesh()
    data.add_field('class',np.zeros(data.element_count(level),dtype=np.int32),FieldType.SEMANTIC,
                   level=level,schema=schema())
    layer=MeshLayer(data)
    layer.set_active_field(level,'class')
    selected=np.arange(data.element_count(level))[::2]
    layer.active_selection=selected
    ctx=ToolContext(SimpleNamespace(active_layer=layer),Mock(),Mock())
    manager=CommandManager()
    tool=MeshAnnotateTool(manager,ctx)
    parent=QWidget()
    widget=tool.create_context_widget(parent)
    tool.values.setCurrentIndex(1)
    tool._accept()
    np.testing.assert_array_equal(data.get_fields(names='class',level=level)[0].data.ravel()[selected],1)
    # A subsequent field-editor undo may restore different FieldBase objects.
    data.field_container(level)._fields=deepcopy(data.field_container(level)._fields)
    manager.undo()
    np.testing.assert_array_equal(data.get_fields(names='class',level=level)[0].data.ravel(),0)
    manager.redo()
    np.testing.assert_array_equal(data.get_fields(names='class',level=level)[0].data.ravel()[selected],1)


def test_deleting_preview_cancels_without_touching_document(app):
    from geon.ui.scene_manager import SceneManager
    tool,ctx,window,widget=preview_setup(app)
    fake=SimpleNamespace(tool_controller=ctx.controller,_scene=ctx.scene)
    SceneManager._delete_layer(fake,tool.preview_layer)
    assert tool.preview_layer is None
    assert len(ctx.scene.layers)==1
    assert len(ctx.scene.doc.scene_items)==1
    assert tool.source.visible
    window.close()
