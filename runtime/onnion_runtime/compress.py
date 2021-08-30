import numpy as np


class Compress:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis")

    def run(self, x, cond):
        return [np.compress(cond, x, axis=self.axis)]
