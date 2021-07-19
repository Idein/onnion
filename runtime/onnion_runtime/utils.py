import numpy as np


# Ref: https://github.com/onnx/onnx/blob/bf426626378cbb715d3d0000a35f065708156cbc/onnx/onnx.proto#L266-L287
def tensor_type_to_dtype(t: int) -> np.dtype:
    mapping = [
        None,  # UNDEFINED
        np.float32,
        np.uint8,
        np.int8,
        np.uint16,
        np.int16,
        np.int32,
        np.int64,
        np.object,
        np.bool,
        np.float16,
        np.float64,
        np.uint32,
        np.uint64,
        np.complex64,
        np.complex128,
    ]

    return mapping[t]
