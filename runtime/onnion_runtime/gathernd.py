import numpy as np


class GatherND:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.batch_dims = kwargs.get("batch_dims", 0)

    def run(self, data, indices):
        return [gather_nd_impl(data, indices, self.batch_dims)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/db1a9f2388bd48e0bdde095f231a4dcc1473430a/onnx/backend/test/case/node/gathernd.py#L15-L54
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def gather_nd_impl(data, indices, batch_dims):
    # type: (np.ndarray, np.ndarray, int) -> np.ndarray
    # Note the data rank - will be reused multiple times later
    data_rank = len(data.shape)

    # Check input tensors' shape/rank condition
    assert indices.shape[-1] <= data_rank

    # The list of data/indice shape of batch_dims
    batch_dims_shape = []

    # The number of elements in the batch_dims for data/indice array
    batch_dims_size = 1

    # Check the shape of indice and data are identicial for batch dims.
    for i in range(batch_dims):
        batch_dims_shape.append(indices.shape[i])
        batch_dims_size *= indices.shape[i]

    # Compute output of the op as below

    # Compute shape of output array
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape + list(indices.shape)[batch_dims:-1] + list(data.shape)[batch_dims + indices.shape[-1] :]
    )

    # Placeholder for output data
    output_data_buffer = []

    # Flatten 'indices' to 2D array
    reshaped_indices = indices.reshape(batch_dims_size, -1, indices.shape[-1])

    # Flatten 'data' to array of shape (batch_dim_size, data.shape[batch_dimes:])
    reshaped_data = data.reshape((batch_dims_size,) + data.shape[batch_dims:])

    # gather each scalar value from 'data'
    for batch_dim in range(reshaped_indices.shape[0]):
        for outer_dim in range(reshaped_indices.shape[1]):
            gather_index = tuple(reshaped_indices[batch_dim][outer_dim])
            output_data_buffer.append(reshaped_data[(batch_dim,) + gather_index])
    return np.asarray(output_data_buffer, dtype=data.dtype).reshape(output_shape)
