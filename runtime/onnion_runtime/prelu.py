import numpy as np


class PRelu:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x, slope):
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope
        return [y]
