import numpy as np

from .error import RunError


class Min:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, *xs):
        if len(xs) < 1:
            raise RunError("Min", self.version)
        else:
            acc = xs[0]
            for x in xs[1:]:
                acc = np.minimum(acc, x)
            return [acc]
