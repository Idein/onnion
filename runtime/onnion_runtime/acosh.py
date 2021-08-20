import numpy as np


class Acosh:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        return [np.arccosh(x)]
