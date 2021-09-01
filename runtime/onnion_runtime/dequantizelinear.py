import numpy as np


class DequantizeLinear:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", 1)

    def run(self, x, x_scale, x_zero_point=None):
        if x_zero_point is None:
            z = np.zeros(x_scale.size).reshape(x_scale.shape).astype(np.uint8)
        else:
            z = x_zero_point

        if x_scale.size == 1:
            y = (x.astype(np.float32) - z.astype(np.float32)) * x_scale
            return [y]

        else:
            sh = np.ones(len(x.shape)).astype(int)
            sh[self.axis] = x_scale.size
            y = (x.astype(np.float32) - z.reshape(sh).astype(np.float32)) * x_scale.reshape(sh)
            return [y]
