import numpy as np

from .error import RunError


class Or:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis")
        self.broadcast = kwargs.get("broadcast", 0)

    def run(self, x, y):
        if self.version >= 7:
            return [np.logical_or(x, y)]
        else:
            raise RunError("Or", self.version)
