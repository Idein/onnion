import numpy as np


class Not:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        return [np.logical_not(x)]
