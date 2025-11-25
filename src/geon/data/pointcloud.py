import numpy as np
from .registry import register_data
from .base import BaseData

@register_data
class PointCloudData(BaseData):
    def __init__(self, points: np.ndarray):
        self.points = points
        self.segmentations = dict[SegmentationBase]