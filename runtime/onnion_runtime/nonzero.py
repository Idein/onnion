import numpy as np


class NonZero:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        y = np.array(np.nonzero(x), dtype=np.int64)
        return [y]
