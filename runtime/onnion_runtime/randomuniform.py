import numpy as np

from .utils import tensor_type_to_dtype


class RandomUniform:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.dtype = tensor_type_to_dtype(kwargs.get("dtype", 1))
        self.high = kwargs.get("high", 1.0)
        self.low = kwargs.get("low", 0.0)
        self.seed = kwargs.get("seed")
        self.shape = kwargs["shape"]

    def run(self):
        if self.seed is not None:
            np.random.seed(int(self.seed))

        x = np.random.uniform(self.low, self.high, tuple(self.shape))
        return [x.astype(self.dtype)]

    def warning(opset_version):
        return "RandomUniform does not return the same output as onnxruntime even if set seed."
