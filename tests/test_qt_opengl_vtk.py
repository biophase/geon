"""Requires a real Qt OpenGL surface: GEON_TEST_QT_OPENGL=1 QT_QPA_PLATFORM=wayland.

Run separately from offscreen Qt tests, since QApplication's platform is process-wide.
"""
import os

import numpy as np
import pytest
import vtk
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from geon.data.pointcloud import PointCloudData, FieldType
from geon.rendering.pointcloud import PointCloudLayer
from geon.ui.picker import PointPicker
from geon.ui.qt_opengl_vtk import QtOpenGLVTKWidget, configure_surface_format

pytestmark = pytest.mark.skipif(os.environ.get('GEON_TEST_QT_OPENGL') != '1',
                                reason='requires an accessible desktop OpenGL context')


def test_qt_surface_render_pick_resize_and_cleanup():
    configure_surface_format()
    app = QApplication.instance() or QApplication([])
    widget = QtOpenGLVTKWidget()
    widget.resize(200, 200)
    window = widget.GetRenderWindow()
    renderer = vtk.vtkRenderer()
    window.AddRenderer(renderer)
    points = np.array([[0, 0, 1], [0, 0, 0]], dtype=np.float32)
    data = PointCloudData(points)
    data.add_field('blue', np.tile([0, 0, 255], (2, 1)), FieldType.COLOR)
    layer = PointCloudLayer(data)
    layer.attach(renderer)
    layer.update()
    layer._main_actor.GetProperty().SetPointSize(25)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 5)
    camera.SetFocalPoint(0, 0, 0)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(2)
    camera.SetClippingRange(.1, 10)
    interactor = window.GetInteractor()
    interactor.Initialize()
    picker = PointPicker(renderer)
    widget.show()
    QTest.qWait(250)
    edl = None

    def color():
        image = widget.grabFramebuffer()
        assert not image.isNull()
        return image.pixelColor(image.width() // 2, image.height() // 2).getRgb()[:3]

    def pick():
        width, height = window.GetSize()
        return picker.pick(interactor, width // 2, height // 2)

    try:
        assert widget.isValid()
        assert window.GetClassName() == 'vtkGenericOpenGLRenderWindow'
        np.testing.assert_allclose(color(), [0, 0, 255], atol=2)
        widget.doneCurrent()
        window.Render()  # External callers must reacquire Qt's context.
        assert pick().element_id == 0
        np.testing.assert_allclose(color(), [0, 0, 255], atol=2)  # Never present pick IDs.
        layer.set_visibility_mask(np.array([False, True]))
        window.Render()
        assert pick().element_id == 1
        layer.active_selection = np.array([1], dtype=np.int64)
        layer._selection_shader.selection_timer.stop()
        layer._selection_shader.set_pulse(1)
        window.Render()
        np.testing.assert_allclose(color(), [255, 128, 0], atol=2)
        assert pick().element_id == 1
        edl = vtk.vtkEDLShading()
        edl.SetDelegatePass(vtk.vtkRenderStepsPass())
        renderer.SetPass(edl)
        window.Render()
        assert pick().element_id == 1
        assert renderer.GetPass() is edl
        assert max(color()) > 0
        renderer.SetPass(None)
        edl.ReleaseGraphicsResources(window)
        edl = None
        widget.resize(300, 180)
        QTest.qWait(100)
        assert window.GetSize() == (round(300 * widget.devicePixelRatioF()),
                                    round(180 * widget.devicePixelRatioF()))
        assert pick().element_id == 1
        # Verify reused Qt key translation reaches the VTK interactor.
        keys = []
        interactor.AddObserver('KeyPressEvent', lambda *_: keys.append(interactor.GetKeySym()))
        QTest.keyClick(widget, Qt.Key.Key_A)
        assert keys == ['a']
    finally:
        if edl is not None:
            renderer.SetPass(None)
            edl.ReleaseGraphicsResources(window)
        layer.detach()
        widget.close()
        app.processEvents()
    assert not window.GetReadyForRendering()


def test_geon_viewer_native_surface(monkeypatch):
    from geon.ui.viewer import VTKViewer
    monkeypatch.setenv('GEON_QT_RENDERER', 'opengl')
    configure_surface_format()
    app = QApplication.instance() or QApplication([])
    viewer = VTKViewer()
    viewer.resize(320, 240)
    viewer.show()
    QTest.qWait(150)
    widget = viewer.vtkWidget
    window = widget.GetRenderWindow()
    source = vtk.vtkSphereSource()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    viewer._renderer.AddActor(actor)
    viewer._renderer.ResetCamera()
    try:
        viewer.rerender()
        assert widget.isValid()
        camera = viewer._renderer.GetActiveCamera()
        before = camera.GetPosition()
        QTest.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(120, 100))
        QTest.mouseMove(widget, QPoint(170, 130), delay=30)
        QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(170, 130))
        assert camera.GetPosition() != before
        viewer.enable_edl()
        viewer.rerender()
        viewer.disable_edl()
        assert viewer._renderer.GetPass() is None
        assert not widget.grabFramebuffer().isNull()
    finally:
        widget.close()
        viewer.close()
        app.processEvents()
