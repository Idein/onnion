import numpy as np


class Concat:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs["axis"]

    def run(self, *inputs):
        return [np.concatenate(inputs, axis=self.axis)]
