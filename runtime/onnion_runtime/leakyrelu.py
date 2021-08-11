import numpy as np


class LeakyRelu:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.alpha = kwargs.get("alpha", 0.01)
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x):
        y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * self.alpha
        return [y]
