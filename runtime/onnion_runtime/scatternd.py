import numpy as np


class ScatterND:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.reduction = kwargs.get("reduction", "none")

    def run(self, data, indices, updates):
        return [scatter_nd_impl(data, indices, updates, self.reduction)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/07c494bf077e9e4a7898119f28a50585469ad4cd/onnx/backend/test/case/node/scatternd.py#L15-L31
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def scatter_nd_impl(data, indices, updates, reduction="none"):  # type: ignore

    # Check tensor shapes
    assert indices.shape[-1] <= len(data.shape)
    assert updates.shape == indices.shape[:-1] + data.shape[indices.shape[-1] :]

    # Compute output
    output = np.copy(data)
    for i in np.ndindex(indices.shape[:-1]):
        # NOTE: The order of iteration in this loop is not specified.
        if reduction == "add":
            output[indices[i]] += updates[i]
        elif reduction == "mul":
            output[indices[i]] *= updates[i]
        else:
            output[indices[i]] = updates[i]
    return output
