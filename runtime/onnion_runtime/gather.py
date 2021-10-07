import numpy as np


class Gather:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", 0)

    def run(self, data, indices):
        return [np.take(data, indices.astype(int), axis=self.axis)]

    def warning(opset_version):
        return "Gather may not work with raspi as specified by ONNX. It uses int instead of np.int64."
