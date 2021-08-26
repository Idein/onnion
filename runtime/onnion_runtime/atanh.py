import numpy as np


class Atanh:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        return [np.arctanh(x)]
