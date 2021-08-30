import numpy as np


class ConcatFromSequence:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs["axis"]
        self.new_axis = kwargs.get("new_axis", 0)

    def run(self, inputs):
        if self.new_axis == 0:
            return [np.concatenate(inputs, axis=self.axis)]
        else:
            return [np.stack(inputs, axis=self.axis)]
