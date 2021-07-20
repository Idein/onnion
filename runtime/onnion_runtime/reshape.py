import numpy as np

from .error import RunError


class Reshape:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.allowzero = kwargs.get("allowzero", 0)
        self.consumed_inputs = kwargs.get("consumed_inputs")
        self.shape = kwargs.get("shape")

    def run(self, data, shape=None):
        if self.version >= 5:
            self.shape = shape

        if self.shape is None:
            raise RunError("Reshape", self.version)

        return [reshape_reference_implementation(data, self.shape, allowzero=self.allowzero)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/a71135c16c6c8f95fe3b0df2144955f872cb7cda/onnx/backend/test/case/node/reshape.py#L15.
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def reshape_reference_implementation(data, shape, allowzero=0):  # type: (np.ndarray, np.ndarray, int) -> np.ndarray
    # replace zeros with corresponding dim size
    # we need to do this because np.reshape doesn't support 0 by default unless 'allowzero' is set
    new_shape = np.copy(shape)
    if allowzero == 0:
        zeros_index = np.where(shape == 0)
        new_shape[zeros_index] = np.array(data.shape)[zeros_index]
    reshaped = np.reshape(data, new_shape)
    return reshaped
