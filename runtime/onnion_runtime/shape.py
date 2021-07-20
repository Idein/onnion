import numpy as np


class Shape:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        return [np.array(x.shape).astype(np.int64)]
