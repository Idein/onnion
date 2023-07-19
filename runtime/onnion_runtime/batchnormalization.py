import numpy as np

from .error import RunError


class BatchNormalization:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.epsilon = kwargs.get("epsilon", 1e-5)
        if 14 <= self.version:
            training_mode = kwargs.get("training_mode", 0)
            if training_mode != 0:
                # Non-zero values for training_mode are not supported
                raise RunError("BatchNormalization", self.version)

        # only opset_versions (>= 9) are supported
        raise RunError("BatchNormalization", self.version)

    def run(self, x, scale, bias, input_mean, input_var):
        """
        Perform batch normalization

        Arguments are same to the original implementation,
        but output is a singleton list wraps the output of the original.

        Returns:
          List of an element which is the ouput tensor of the same shape as x
        """
        return [_batchnormalization(x, scale, bias, input_mean, input_var, self.epsilon)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/13f02aacb5d10495eed55a59729d9d2993db43bf/onnx/backend/test/case/node/batchnorm.py#L12
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def _batchnormalization(x, s, bias, mean, var, epsilon):  # type: ignore
    dims_x = len(x.shape)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    mean = mean.reshape(-1, *dim_ones)
    var = var.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias
