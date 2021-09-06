import numpy as np


class Einsum:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.equation = kwargs["equation"]

    def run(self, *inputs):
        return [np.einsum(self.equation, *inputs)]
