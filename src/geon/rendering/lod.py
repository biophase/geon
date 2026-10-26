"""Shared viewport budgeting for renderable objects with adjustable detail."""

from time import perf_counter
from contextlib import contextmanager
from typing import Protocol

import vtk
import os

# auto | coarse-interaction | coarse-display
LOD_DEBUG_MODE = os.environ.get("GEON_LOD_DEBUG", "auto")

class LODParticipant(Protocol):
    """Levels are ordered from full detail to least detail; costs are seconds."""

    level: int

    @property
    def visible(self) -> bool: ...

    def costs(self) -> list[float]: ...
    def select_level(self, level: int) -> None: ...
    def record_frame(self) -> float: ...


class LODCoordinator:
    """One shared budget per renderer; non-participant work is frame overhead."""

    def __init__(self, renderer):
        self.renderer = renderer
        self.participants: list[LODParticipant] = []
        self.overhead = 0.0
        self._frame = None
        self._picking_depth = 0
        self._start_observer = renderer.AddObserver(vtk.vtkCommand.StartEvent, self._start)
        self._end_observer = renderer.AddObserver(vtk.vtkCommand.EndEvent, self._end)

    @classmethod
    def for_renderer(cls, renderer):
        coordinator = getattr(renderer, '_geon_lod_coordinator', None)
        if coordinator is None:
            coordinator = cls(renderer)
            renderer._geon_lod_coordinator = coordinator
        return coordinator

    def register(self, participant):
        self.participants.append(participant)
        self.overhead = 0.0

    @contextmanager
    def picking(self):
        # Point selection also does a preparatory depth render with no selector
        # attached. Protect the entire operation, not only its ID passes.
        self._picking_depth += 1
        try:
            for participant in self.participants:
                participant.select_level(0)
            yield
        finally:
            self._picking_depth -= 1

    def unregister(self, participant):
        if participant not in self.participants:
            return
        self.participants.remove(participant)
        self.overhead = 0.0
        if not self.participants:
            self.renderer.RemoveObserver(self._start_observer)
            self.renderer.RemoveObserver(self._end_observer)
            del self.renderer._geon_lod_coordinator

    @staticmethod
    def choose_levels(costs, budget, previous):
        """Spend a global budget, reducing the largest cost first.

        A 20% margin before restoring detail prevents fine/coarse oscillation.
        The same policy supports more than two levels per participant.
        """
        levels = [0] * len(costs)
        target = budget * (0.8 if any(previous) else 1.0)
        total = sum(values[0] for values in costs)
        while total > target:
            choices = [
                (values[levels[i]] - values[level], i, level)
                for i, values in enumerate(costs)
                for level in range(levels[i] + 1, len(values))
            ]
            if not choices:
                break
            saving, index, level = max(choices)
            if saving <= 0:
                break
            levels[index] = level
            total -= saving
        return levels

    def _interactive(self):
        window = self.renderer.GetRenderWindow()
        interactor = window.GetInteractor() if window is not None else None
        if interactor is not None:
            # VTK's interaction style sets the window rate on interaction start
            # and restores the still rate before the final full-detail render.
            return window.GetDesiredUpdateRate() > interactor.GetStillUpdateRate()
        # Offscreen/headless callers can supply an explicit interactive budget.
        return self.renderer.GetAllocatedRenderTime() < 1.0

    def _start(self, *_):
        picking = (
            self._picking_depth > 0
            or self.renderer.GetSelector() is not None
        )
        interactive = self._interactive()
        visible = [p for p in self.participants if p.visible]

        force_coarse = (
            LOD_DEBUG_MODE == "coarse-display"
            or (
                LOD_DEBUG_MODE == "coarse-interaction"
                and interactive
            )
        )

        if picking:
            for participant in self.participants:
                participant.select_level(0)

        elif force_coarse:
            for participant in visible:
                participant.select_level(len(participant.costs()) - 1)

        elif not interactive:
            for participant in self.participants:
                participant.select_level(0)

        else:
            costs = [p.costs() for p in visible]
            budget = max(
                0.0,
                self.renderer.GetAllocatedRenderTime() - self.overhead,
            )
            levels = self.choose_levels(
                costs, budget, [p.level for p in visible],
            )
            for participant, level in zip(visible, levels):
                participant.select_level(level)

    def _end(self, *_):
        frame, self._frame = self._frame, None
        if frame is None or self._picking_depth or self.renderer.GetSelector() is not None:
            return
        window = self.renderer.GetRenderWindow()
        if window is not None and window.GetAbortRender():
            return
        started, visible = frame
        elapsed = perf_counter() - started
        measured = sum(p.record_frame() for p in visible)
        # Mapper timings are estimates, not additive GPU timings. Calibrate
        # against the whole display render, including EDL and ordinary props.
        residual = max(0.0, elapsed - measured)
        self.overhead = 0.8 * self.overhead + 0.2 * residual

