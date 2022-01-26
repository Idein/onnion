import numpy as np

from .utils import tensor_type_to_dtype


class RandomNormalLike:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.dtype = kwargs.get("dtype")
        self.mean = kwargs.get("mean", 0.0)
        self.scale = kwargs.get("scale", 1.0)
        self.seed = kwargs.get("seed")

    def run(self, x):
        if self.seed is not None:
            np.random.seed(int(self.seed))

        if self.dtype is None:
            self.dtype = x.dtype
        else:
            self.dtype = tensor_type_to_dtype(self.dtype)

        x = np.random.normal(self.mean, self.scale, tuple(x.shape))
        return [x.astype(self.dtype)]

    def warning(opset_version):
        return "RandomNormalLike does not return the same output as onnxruntime even if set seed."
