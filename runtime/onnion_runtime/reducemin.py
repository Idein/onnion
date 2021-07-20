import numpy as np


class ReduceMin:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axes = kwargs.get("axes")
        self.keepdims = kwargs.get("keepdims", 1)

    def run(self, data):
        if self.axes is None:
            return [np.minimum.reduce(data, axis=self.axes, keepdims=self.keepdims == 1)]
        else:
            return [np.minimum.reduce(data, axis=tuple(self.axes), keepdims=self.keepdims == 1)]
