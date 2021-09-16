import numpy as np


class GreaterOrEqual:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x, y):
        return [np.greater_equal(x, y)]
