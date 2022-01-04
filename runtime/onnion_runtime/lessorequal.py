import numpy as np


class LessOrEqual:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x, y):
        return [np.less_equal(x, y)]
