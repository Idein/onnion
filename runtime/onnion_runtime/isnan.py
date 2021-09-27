import numpy as np


class IsNaN:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        return [np.isnan(x)]
