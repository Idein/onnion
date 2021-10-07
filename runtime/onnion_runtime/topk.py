import numpy as np

from .error import RunError


class TopK:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.k = kwargs.get("k")
        self.axis = kwargs.get("axis", -1)
        self.sorted = kwargs.get("sorted", 1)
        self.largest = kwargs.get("largest", 1)

    def run(self, x, k=None):
        if self.version >= 10:
            self.k = k

        if self.k is None:
            raise RunError("TopK", self.version)

        values, indices = topk_sorted_implementation(x, self.k, self.axis, self.largest)
        return [values, indices]

    def warning(opset_version):
        return "TopK may not work with raspi as specified by ONNX. It uses int instead of np.int64."


# The following code has been copied from
# https://github.com/onnx/onnx/blob/2875f51853133d9cc028bbdccb62f92745cb94c2/onnx/backend/test/case/node/topk.py#L15-L23
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
# Note: Modify np.arange(k) -> np.arange(k).astype(int)
def topk_sorted_implementation(X, k, axis, largest):  # type: ignore
    sorted_indices = np.argsort(X, axis=axis)
    sorted_values = np.sort(X, axis=axis)
    if largest:
        sorted_indices = np.flip(sorted_indices, axis=axis)
        sorted_values = np.flip(sorted_values, axis=axis)
    topk_sorted_indices = np.take(sorted_indices, np.arange(k).astype(int), axis=axis)
    topk_sorted_values = np.take(sorted_values, np.arange(k).astype(int), axis=axis)
    return topk_sorted_values, np.array(topk_sorted_indices, dtype=np.int64)
