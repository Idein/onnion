import numpy as np


class HardSwish:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        return [hardswish(x)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/a5e7ee51176bf78a60c118758174e13d85a87b46/onnx/backend/test/case/node/hardswish.py#L15-L18
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def hardswish(x):  # type: (np.ndarray) -> np.ndarray
    alfa = float(1 / 6)
    beta = 0.5
    return x * np.maximum(0, np.minimum(1, alfa * x + beta))
