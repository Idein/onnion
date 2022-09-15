import numpy as np

from .error import RunError


class Split:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", 0)
        if opset_version < 13:
            self.split = kwargs.get("split")

    def run(self, x, split=None):
        if len(x.shape) <= self.axis:
            raise RunError("Split", self.version)

        if split is None and self.version < 13:
            split = self.split

        if split is None:
            raise RunError("Split", self.version)  # unsuported

        if sum(split) != x.shape[self.axis]:
            raise RunError("Split", self.version)

        if len(split) == 1:
            return [np.copy(x)]
        else:
            indices = np.cumsum(split)[:-1]
            output = np.split(x, indices_or_sections=indices, axis=self.axis)
            assert len(output) == len(split)
            return output
