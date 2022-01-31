import numpy as np


class ReduceLogSum:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axes = kwargs.get("axes")
        self.keepdims = kwargs.get("keepdims", 1)

    def run(self, x):
        if self.axes is not None:
            self.axes = tuple(self.axes)
        return [np.log(np.sum(x, axis=self.axes, keepdims=self.keepdims == 1))]
