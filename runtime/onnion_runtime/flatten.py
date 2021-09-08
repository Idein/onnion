import numpy as np


class Flatten:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", 1)

    def run(self, x):
        shape = x.shape
        assert len(shape) >= self.axis
        new_shape = (np.prod(shape[0 : self.axis]).astype(int), -1)
        return [np.reshape(x, new_shape)]
