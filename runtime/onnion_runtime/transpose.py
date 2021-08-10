import numpy as np


class Transpose:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.perm = kwargs.get("perm")

    def run(self, x):
        y = x.copy()
        return [np.transpose(y, axes=self.perm)]
