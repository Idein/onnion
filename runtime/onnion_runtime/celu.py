import numpy as np


class Celu:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.alpha = kwargs.get("alpha", 1.0)

    def run(self, x):
        y = np.maximum(0, x) + np.minimum(0, self.alpha * (np.exp(x / self.alpha) - 1))
        return [y]
