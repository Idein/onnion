import numpy as np


class Clip:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.consumed_inputs = kwargs.get("consumed_inputs")
        self.max = kwargs.get("max", np.inf)
        self.min = kwargs.get("min", -np.inf)

    def run(self, x, min_val=None, max_val=None):
        if self.version >= 11:
            if min_val is not None:
                self.min = min_val
            if max_val is not None:
                self.max = max_val

        return [np.clip(x, self.min, self.max)]
