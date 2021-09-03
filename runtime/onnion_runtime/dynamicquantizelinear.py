import numpy as np


class DynamicQuantizeLinear:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        x_min = np.minimum(0, np.min(x))
        x_max = np.maximum(0, np.max(x))
        y_scale = np.array((x_max - x_min) / (255 - 0)).astype(np.float32)
        y_zero = np.clip(np.round((0 - x_min) / y_scale), 0, 255).astype(np.uint8)
        y = np.clip(np.round(x / y_scale) + y_zero, 0, 255).astype(np.uint8)

        return [y, y_scale, y_zero]
