import numpy as np

from .error import RunError


class And:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.broadcast = kwargs.get("broadcast", 0)

    def run(self, x, y):
        if self.version >= 7:
            return [np.logical_and(x, y)]
        else:
            raise RunError("And", self.version)
