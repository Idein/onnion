import numpy as np


class Relu:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x):
        return [np.clip(x, 0, np.inf)]
