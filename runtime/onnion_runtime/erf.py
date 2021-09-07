import math

import numpy as np


class Erf:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        y = np.vectorize(math.erf)(x).astype(x.dtype)
        return [y]
