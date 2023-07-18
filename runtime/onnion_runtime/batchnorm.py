import numpy as np

class BatchNormalization:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        assert self.version >= 15, f"Now, only opset_versions (>= 15) are supported"
        self.epsilon = kwargs.get("epsilon", 1e-5)
        assert 0 == kwargs.get("training_mode", 0), 'Non-zero values for training_mode are not supported'

    def run(self, x, scale, bias, input_mean, input_var):
        """
        Perform batch normalization

        This implementation is copied from onnion test code:
        https://github.com/onnx/onnx/blob/13f02aacb5d10495eed55a59729d9d2993db43bf/onnx/backend/test/case/node/batchnorm.py#L12
        Arguments are same to the original implementation,
        but output is a singleton list wraps the output of the original.

        Returns:
          List of an element which is the ouput tensor of the same shape as x
        """
        dims_x = len(x.shape)
        dim_ones = (1,) * (dims_x - 2)
        scale = scale.reshape(-1, *dim_ones)
        bias = bias.reshape(-1, *dim_ones)
        mean = input_mean.reshape(-1, *dim_ones)
        var = input_var.reshape(-1, *dim_ones)
        return [scale * (x - mean) / np.sqrt(var + self.epsilon) + bias]

