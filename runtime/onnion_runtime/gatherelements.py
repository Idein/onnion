import numpy as np


class GatherElements:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis", 0)

    def run(self, data, indices):
        return [gather_elements(data, indices, self.axis)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/db1a9f2388bd48e0bdde095f231a4dcc1473430a/onnx/backend/test/case/node/gatherelements.py#L16-L21
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def gather_elements(data, indices, axis=0):  # type: ignore
    data_swaped = np.swapaxes(data, 0, axis)
    index_swaped = np.swapaxes(indices, 0, axis)
    gathered = np.choose(index_swaped, data_swaped, mode="wrap")
    y = np.swapaxes(gathered, 0, axis)
    return y
