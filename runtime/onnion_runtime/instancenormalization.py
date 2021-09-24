import numpy as np


class InstanceNormalization:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.epsilon = kwargs.get("epsilon", 1e-5)
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x, scale, bias):
        return [_instancenorm_test_mode(x, scale, bias, epsilon=self.epsilon)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/8f441d4ec9f4f146010e8c2f6da23b12783909d5/onnx/backend/test/case/node/instancenorm.py#L19-L27
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    axis = tuple(range(2, dims_x))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias
