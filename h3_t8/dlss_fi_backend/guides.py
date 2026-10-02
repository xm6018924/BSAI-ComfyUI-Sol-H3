"""CPU motion guidance only; no image warp and no generated RGB fallback."""
from __future__ import annotations

import numpy as np


class MotionGuide:
    def __init__(self, width, height):
        import cv2
        if type(width) is not int or type(height) is not int or min(width, height) < 64 or width*height > 4096*2160:
            raise ValueError("Motion guide requires a bounded canvas with both axes >=64")
        self.width, self.height = width, height
        ratio = min(1., 640/width)
        self.small = (max(64, round(width*ratio/2)*2), max(64, round(height*ratio/2)*2))
        self.estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.estimator.setFinestScale(1)
        self.estimator.setUseSpatialPropagation(True)
        self.previous = None

    def process(self, rgba, *, reset=False):
        import cv2
        if type(reset) is not bool or not isinstance(rgba, np.ndarray) or rgba.dtype != np.uint8 or rgba.shape != (self.height, self.width, 4):
            raise ValueError("Guide requires exact uint8 RGBA geometry and explicit boolean reset")
        current = cv2.resize(cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY), self.small, interpolation=cv2.INTER_AREA)
        first = self.previous is None
        if first or reset or np.array_equal(current, self.previous):
            field = np.zeros((self.height, self.width, 2), dtype=np.float16)
        else:
            # Backward displacement: where this current pixel was in the previous
            # input. Resize the vector field, then rescale its pixel units.
            field = cv2.resize(self.estimator.calc(current, self.previous, None), (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            field *= np.array([self.width/self.small[0], self.height/self.small[1]], dtype=np.float32)
            if not bool(np.isfinite(field).all()) or float(np.max(np.abs(field))) > np.finfo(np.float16).max:
                raise ValueError("Invalid optical-flow guide; not silently reset/copied as a successful interval")
            field = np.ascontiguousarray(field, dtype=np.float16)
        self.previous = current
        return field, first or reset
