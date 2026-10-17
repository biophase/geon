"""Experimental Qt-owned OpenGL surface for VTK, including native Wayland.

Opt in with GEON_QT_RENDERER=opengl. No native window IDs are passed to VTK.
The framebuffer lifecycle follows VTK's QVTKRenderWindowAdapter: normal renders
publish a display framebuffer, picking renders do not, and Qt presents it.
"""

from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QOpenGLContext, QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk


def configure_surface_format():
    """Call before QApplication creates any OpenGL contexts."""
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(3, 2)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(0)
    fmt.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(fmt)


class QtOpenGLVTKWidget(QOpenGLWidget):
    # Reuse VTK's Python input translation, not its window construction/painting.
    _CURSOR_MAP = QVTKRenderWindowInteractor._CURSOR_MAP
    _GetKeyCharAndKeySym = QVTKRenderWindowInteractor._GetKeyCharAndKeySym
    _GetCtrlShift = QVTKRenderWindowInteractor._GetCtrlShift
    _getPixelRatio = staticmethod(QVTKRenderWindowInteractor._getPixelRatio)
    _setEventInformation = QVTKRenderWindowInteractor._setEventInformation
    enterEvent = QVTKRenderWindowInteractor.enterEvent
    leaveEvent = QVTKRenderWindowInteractor.leaveEvent
    mousePressEvent = QVTKRenderWindowInteractor.mousePressEvent
    mouseReleaseEvent = QVTKRenderWindowInteractor.mouseReleaseEvent
    mouseMoveEvent = QVTKRenderWindowInteractor.mouseMoveEvent
    keyPressEvent = QVTKRenderWindowInteractor.keyPressEvent
    keyReleaseEvent = QVTKRenderWindowInteractor.keyReleaseEvent
    wheelEvent = QVTKRenderWindowInteractor.wheelEvent
    CreateTimer = QVTKRenderWindowInteractor.CreateTimer
    DestroyTimer = QVTKRenderWindowInteractor.DestroyTimer
    TimerEvent = QVTKRenderWindowInteractor.TimerEvent
    CursorChangedEvent = QVTKRenderWindowInteractor.CursorChangedEvent
    ShowCursor = QVTKRenderWindowInteractor.ShowCursor
    HideCursor = QVTKRenderWindowInteractor.HideCursor

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ActiveButton = Qt.MouseButton.NoButton
        # The reused event handlers refer to these name-mangled attributes.
        self._QVTKRenderWindowInteractor__saveX = 0
        self._QVTKRenderWindowInteractor__saveY = 0
        self._QVTKRenderWindowInteractor__saveModifiers = Qt.KeyboardModifier.NoModifier
        self._QVTKRenderWindowInteractor__saveButtons = Qt.MouseButton.NoButton
        self._QVTKRenderWindowInteractor__wheelDelta = 0
        self._RenderWindow = vtk.vtkGenericOpenGLRenderWindow()
        self._RenderWindow.SetReadyForRendering(False)
        self._RenderWindow.SetOwnContext(False)
        self._RenderWindow.SetFrameBlitModeToNoBlit()
        self._RenderWindow.SetMultiSamples(0)
        self._Iren = vtk.vtkGenericRenderWindowInteractor()
        self._Iren.SetRenderWindow(self._RenderWindow)
        self._Timer = QTimer(self)
        self._Timer.timeout.connect(self.TimerEvent)
        self._Iren.AddObserver('CreateTimerEvent', self.CreateTimer)
        self._Iren.AddObserver('DestroyTimerEvent', self.DestroyTimer)
        for event, callback in (
            ('WindowMakeCurrentEvent', self._make_current),
            ('WindowIsCurrentEvent', self._is_current),
            ('WindowFrameEvent', self._frame),
            ('CursorChangedEvent', self.CursorChangedEvent),
        ):
            self._RenderWindow.AddObserver(event, callback)
        self._needs_render = True
        self._in_paint = False
        self._surface = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)
        if parent is not None:
            parent.destroyed.connect(self.close)

    def GetRenderWindow(self):
        return self._RenderWindow

    def Render(self):
        self._RenderWindow.Render()

    def sizeHint(self):
        return QSize(400, 400)

    def _make_current(self, *_):
        # QOpenGLWidget.makeCurrent also binds Qt's framebuffer. During a VTK
        # render, that would replace VTK's currently bound render/pick target.
        context = self.context()
        if context is not None and self._surface is not None:
            if QOpenGLContext.currentContext() != context or context.surface() != self._surface:
                context.makeCurrent(self._surface)

    def _is_current(self, *_):
        self._RenderWindow.SetIsCurrent(
            self.context() is not None and QOpenGLContext.currentContext() == self.context())

    def initializeGL(self):
        self._surface = self.context().surface()
        self.context().aboutToBeDestroyed.connect(self._cleanup)
        window = self._RenderWindow
        window.SetReadyForRendering(True)
        window.SetIsCurrent(True)
        window.OpenGLInitContext()
        self._needs_render = True

    def resizeGL(self, width, height):
        scale = self.devicePixelRatioF()
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
        self._RenderWindow.SetDPI(round(72 * scale))
        self._Iren.UpdateSize(width, height)
        self._Iren.ConfigureEvent()
        self._needs_render = True

    def _frame(self, *_):
        window = self._RenderWindow
        if window.GetDoubleBuffer() and not window.GetSwapBuffers():
            return  # Selection buffers must never be presented to the user.
        self._needs_render = False
        if not self._in_paint:
            self.update()

    def paintGL(self):
        self._in_paint = True
        try:
            window = self._RenderWindow
            if self._needs_render:
                window.Render()
                self._needs_render = False
            state = window.GetState()
            state.Reset()
            state.Push()
            try:
                state.vtkglDisable(0x0C11)  # GL_SCISSOR_TEST
                state.vtkglBindFramebuffer(0x8CA9, self.defaultFramebufferObject())
                width, height = window.GetSize()
                window.BlitDisplayFramebuffer(
                    0, 0, 0, width, height, 0, 0, width, height,
                    0x00004000, 0x2600)  # GL_COLOR_BUFFER_BIT, GL_NEAREST
                # Qt composites the widget; don't blend VTK's background alpha
                # with widgets behind it. Match QVTKRenderWindowAdapter::clearAlpha.
                state.vtkglColorMask(False, False, False, True)
                state.vtkglClearColor(0., 0., 0., 1.)
                state.vtkglClear(0x00004000)
            finally:
                state.Pop()
        finally:
            self._in_paint = False

    def _cleanup(self):
        if not self._RenderWindow.GetReadyForRendering():
            return
        self._Timer.stop()
        self.makeCurrent()
        renderers = self._RenderWindow.GetRenderers()
        renderers.InitTraversal()
        for _ in range(renderers.GetNumberOfItems()):
            renderer = renderers.GetNextItem()
            display_pass = renderer.GetPass()
            if display_pass is not None:
                display_pass.ReleaseGraphicsResources(self._RenderWindow)
        self._RenderWindow.Finalize()
        self._RenderWindow.SetReadyForRendering(False)
        self.doneCurrent()
        self._surface = None

    def closeEvent(self, event):
        self._cleanup()
        super().closeEvent(event)
