import numpy as np

from .utils import tensor_type_to_dtype


class RandomUniformLike:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.dtype = kwargs.get("dtype")
        self.high = kwargs.get("high", 1.0)
        self.low = kwargs.get("low", 0.0)
        self.seed = kwargs.get("seed")

    def run(self, x):
        if self.seed is not None:
            np.random.seed(int(self.seed))

        if self.dtype is None:
            self.dtype = x.dtype
        else:
            self.dtype = tensor_type_to_dtype(self.dtype)

        x = np.random.uniform(self.low, self.high, tuple(x.shape))
        return [x.astype(self.dtype)]

    def warning(opset_version):
        return "RandomUniformLike does not return the same output as onnxruntime even if set seed."
