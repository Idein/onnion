import numpy as np

from .error import RunError


class Unsqueeze:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axes = kwargs.get("axes")

    def run(self, data, axes=None):
        if self.version >= 13:
            self.axes = axes

        if self.axes is None:
            raise RunError("Unsqueeze", self.version)

        result = data.copy()
        for a in sorted(self.axes):
            result = np.expand_dims(result, axis=a)

        return [result]
