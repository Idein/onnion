import numpy as np


class MatMul:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x, y):
        return [np.matmul(x, y)]
