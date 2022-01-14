import numpy as np


class MatMulInteger:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x, y, x_zero_point=None, y_zero_point=None):
        a = x.astype(np.int32)
        b = y.astype(np.int32)

        if x_zero_point is not None:
            a = a - x_zero_point.astype(np.int32)

        if y_zero_point is not None:
            b = b - y_zero_point.astype(np.int32)

        return [np.matmul(a, b).astype(np.int32)]
