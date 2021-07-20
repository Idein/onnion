import numpy as np


class Squeeze:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axes = kwargs.get("axes")

    def run(self, data, axes=None):
        if self.version >= 13:
            self.axes = axes

        if self.axes is None:
            return [np.squeeze(data, axis=self.axes)]
        else:
            return [np.squeeze(data, axis=tuple(self.axes))]
