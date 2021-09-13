import numpy as np


class Where:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, cond, x, y):
        return [np.where(cond, x, y)]
