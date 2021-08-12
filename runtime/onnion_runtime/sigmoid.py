import numpy as np


class Sigmoid:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x):
        return [1.0 / (1.0 + np.exp(np.negative(x)))]
