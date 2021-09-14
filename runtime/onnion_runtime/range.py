import numpy as np


class Range:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, start, limit, delta):
        y = np.arange(start, limit, delta).astype(start.dtype)
        return [y]
