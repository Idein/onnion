import numpy as np

from .error import RunError


class Split:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", 0)

    def run(self, x, split=None):
        # TODO: check version
        if len(x.shape) <= self.axis:
            raise RunError("Split", self.version)

        if split is None:
            raise RunError("Split", self.version)  # unsuported

        if sum(split) != x.shape[self.axis]:
            raise RunError("Split", self.version)  # unsuported

        if len(split) == 1:
            return [x]
        else:
            indices = np.cumsum(split)[:-1]
            output = np.split(x, indices_or_sections=indices)
            assert len(output) == len(split)
            return output
