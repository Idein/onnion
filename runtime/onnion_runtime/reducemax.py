import numpy as np


class ReduceMax:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axes = kwargs.get("axes")
        self.keepdims = kwargs.get("keepdims", 1)

    def run(self, data):
        if self.axes is None:
            return [np.maximum.reduce(data, axis=self.axes, keepdims=self.keepdims == 1)]
        else:
            return [np.maximum.reduce(data, axis=tuple(self.axes), keepdims=self.keepdims == 1)]
