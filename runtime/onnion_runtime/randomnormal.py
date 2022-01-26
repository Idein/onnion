import numpy as np

from .utils import tensor_type_to_dtype


class RandomNormal:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.dtype = tensor_type_to_dtype(kwargs.get("dtype", 1))
        self.mean = kwargs.get("mean", 0.0)
        self.scale = kwargs.get("scale", 1.0)
        self.seed = kwargs.get("seed")
        self.shape = kwargs["shape"]

    def run(self):
        if self.seed is not None:
            np.random.seed(int(self.seed))

        x = np.random.normal(self.mean, self.scale, tuple(self.shape))
        return [x.astype(self.dtype)]

    def warning(opset_version):
        return "RandomNormal does not return the same output as onnxruntime even if set seed."
