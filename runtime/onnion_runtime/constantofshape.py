import numpy as np


class ConstantOfShape:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.value = kwargs.get("value")

    def run(self, shape):
        n = np.prod(shape)
        if self.value is None:
            return [np.zeros(n).astype(np.float32).reshape(shape)]
        else:
            return [np.repeat(self.value, n).astype(self.value.dtype).reshape(shape)]
