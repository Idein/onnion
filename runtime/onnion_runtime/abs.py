import numpy as np


class Abs:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x):
        return [np.abs(x)]
