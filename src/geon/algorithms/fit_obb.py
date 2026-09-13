from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from geon.data.boundingbox import BoundingBox


FitObbMethod = Literal["pca", "pca_z_locked"]


def _canonical_direction(direction: NDArray[np.float64]) -> NDArray[np.float64]:
    """Resolve the arbitrary sign of an eigenvector deterministically."""
    result = np.asarray(direction, dtype=np.float64).copy()
    dominant = int(np.argmax(np.abs(result)))
    if result[dominant] < 0.0:
        result *= -1.0
    return result


def _rotation_to_euler_zyx(rotation: NDArray[np.float64]) -> tuple[float, float, float]:
    """Return yaw, pitch, roll for R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    sin_pitch = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    if abs(math.cos(pitch)) > 1e-10:
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    else:
        # At gimbal lock, choose the deterministic representation roll = 0.
        yaw = math.atan2(float(-rotation[0, 1]), float(rotation[1, 1]))
        roll = 0.0
    return yaw, pitch, roll


def _pca_axes(points: NDArray[np.float64], method: FitObbMethod) -> NDArray[np.float64]:
    centered = points - points.mean(axis=0)
    if method == "pca_z_locked":
        covariance_xy = centered[:, :2].T @ centered[:, :2] / float(points.shape[0])
        _values, vectors = np.linalg.eigh(covariance_xy)
        axis_xy = _canonical_direction(vectors[:, -1])
        axis_x = np.asarray((axis_xy[0], axis_xy[1], 0.0), dtype=np.float64)
        axis_z = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        axis_y = np.cross(axis_z, axis_x)
        return np.column_stack((axis_x, axis_y, axis_z))

    covariance = centered.T @ centered / float(points.shape[0])
    _values, vectors = np.linalg.eigh(covariance)
    axis_x = _canonical_direction(vectors[:, -1])
    axis_z_candidate = _canonical_direction(vectors[:, 0])
    axis_y = np.cross(axis_z_candidate, axis_x)
    axis_y_norm = float(np.linalg.norm(axis_y))
    if axis_y_norm <= 1e-12:  # defensive fallback for numerical degeneracy
        axis_y = _canonical_direction(vectors[:, 1])
    else:
        axis_y /= axis_y_norm
    axis_z = np.cross(axis_x, axis_y)
    axis_z /= max(float(np.linalg.norm(axis_z)), 1e-12)
    return np.column_stack((axis_x, axis_y, axis_z))


def fit_obb(
    points: NDArray[np.floating],
    method: FitObbMethod | str,
    trim: float = 0.01,
) -> BoundingBox:
    """Fit a percentile-trimmed PCA oriented bounding box to ``points``."""
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {point_array.shape}")
    if point_array.shape[0] == 0:
        raise ValueError("At least one point is required to fit a bounding box")
    if not np.all(np.isfinite(point_array)):
        raise ValueError("points must contain only finite coordinates")
    if method not in ("pca", "pca_z_locked"):
        raise ValueError("method must be either 'pca' or 'pca_z_locked'")
    trim_value = float(trim)
    if not np.isfinite(trim_value) or not 0.0 <= trim_value < 0.5:
        raise ValueError("trim must be finite and satisfy 0 <= trim < 0.5")

    rotation = _pca_axes(point_array, method)  # local axes are matrix columns
    projections = point_array @ rotation
    lower = np.percentile(projections, 100.0 * trim_value, axis=0)
    upper = np.percentile(projections, 100.0 * (1.0 - trim_value), axis=0)
    dimensions = np.maximum(upper - lower, 1e-6)

    # BoundingBox uses the center of its bottom face as its local origin.
    local_bottom_center = np.asarray(
        ((lower[0] + upper[0]) * 0.5, (lower[1] + upper[1]) * 0.5, lower[2]),
        dtype=np.float64,
    )
    center_bottom = rotation @ local_bottom_center
    yaw, pitch, roll = _rotation_to_euler_zyx(rotation)
    return BoundingBox(
        center_bottom_xyz=tuple(float(value) for value in center_bottom),
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        width=float(dimensions[0]),
        depth=float(dimensions[1]),
        height=float(dimensions[2]),
        attributes={"fit_method": str(method), "trim": trim_value},
    )
