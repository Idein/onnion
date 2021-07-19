import numpy as np


class Gather:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", 0)

    def run(self, data, indices):
        return [np.take(data, indices, axis=self.axis)]
