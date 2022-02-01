import numpy as np

from .error import RunError


class ReduceSum:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axes = kwargs.get("axes")
        self.keepdims = kwargs.get("keepdims", 1)
        self.noop_with_empty_axes = kwargs.get("noop_with_empty_axes", 0)

    def run(self, x, axes=None):
        if self.version >= 13:
            if axes is None:
                if self.noop_with_empty_axes != 0:
                    # Undefined behavior (maybe).
                    # onnxruntime raise error (show test_reducesum_08).
                    # So onnion also raise error.
                    raise RunError("ReduceSum", self.version)
                else:
                    return [np.sum(x, axis=axes, keepdims=self.keepdims == 1)]
            else:
                if len(axes) == 0:
                    if self.noop_with_empty_axes == 0:
                        return [np.sum(x, axis=None, keepdims=self.keepdims == 1)]
                    else:
                        return [x]
                else:
                    return [np.sum(x, axis=tuple(axes), keepdims=self.keepdims == 1)]
        else:
            if self.axes is not None:
                self.axes = tuple(self.axes)
            return [np.sum(x, axis=self.axes, keepdims=self.keepdims == 1)]
