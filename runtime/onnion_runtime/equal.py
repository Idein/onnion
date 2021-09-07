import numpy as np

from .error import RunError


class Equal:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.broadcast = kwargs.get("broadcast", 0)

    def run(self, x, y):
        if self.version > 6:
            return [np.equal(x, y)]
        else:
            raise RunError("Equal", self.version)
