import numpy as np


class Elu:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.alpha = kwargs.get("alpha", 1.0)
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x):
        y = np.clip(x, 0, np.inf) + self.alpha * (np.exp(np.clip(x, -np.inf, 0)) - 1)
        return [y]
