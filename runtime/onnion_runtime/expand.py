import numpy as np


class Expand:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x, shape):
        return [x * np.ones(shape).astype(x.dtype)]
