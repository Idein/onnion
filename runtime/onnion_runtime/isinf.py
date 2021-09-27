import numpy as np


class IsInf:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.detect_negative = kwargs.get("detect_negative", 1)
        self.detect_positive = kwargs.get("detect_positive", 1)

    def run(self, x):
        if self.detect_negative == 0:
            return [np.isposinf(x)]

        if self.detect_positive == 0:
            return [np.isneginf(x)]

        return [np.isinf(x)]
