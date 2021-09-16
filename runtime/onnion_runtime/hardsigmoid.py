import numpy as np


class HardSigmoid:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.alpha = kwargs.get("alpha", 0.2)
        self.beta = kwargs.get("beta", 0.5)
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x):
        return [np.clip(x * self.alpha + self.beta, 0, 1)]
