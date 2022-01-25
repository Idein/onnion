import numpy as np

from .error import RunError


class Pow:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis")
        self.broadcast = kwargs.get("broadcast")

    def run(self, x, y):
        if self.version > 6:
            z = np.power(x, y)
            return [z.astype(x.dtype)]
        else:
            raise RunError("Pow", self.version)
