import numpy as np

from .utils import tensor_type_to_dtype


class EyeLike:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.dtype = kwargs.get("dtype")
        self.k = kwargs.get("k", 0)

    def run(self, x):
        sh = x.shape
        assert len(sh) == 2

        if self.dtype is None:
            dtype = x.dtype
        else:
            dtype = tensor_type_to_dtype(self.dtype)

        return [np.eye(sh[0], sh[1], k=self.k, dtype=dtype)]
