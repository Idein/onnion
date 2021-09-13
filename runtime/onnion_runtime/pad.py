import numpy as np

from .error import RunError


class Pad:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.mode = kwargs.get("mode", "constant")
        self.pads = kwargs.get("pads")
        self.paddings = kwargs.get("paddings")
        self.value = kwargs.get("value", 0.0)

    def run(self, data, pads=None, constant_value=None):
        if self.version == 1:
            if self.paddings is None:
                raise RunError("Pad", self.version)
            pads = self.paddings
            constant_value = self.value
        elif self.version < 11:
            if self.pads is None:
                raise RunError("Pad", self.version)
            pads = self.pads
            constant_value = self.value
        else:
            if pads is None:
                raise RunError("Pad", self.version)
            if constant_value is None:
                constant_value = 0.0

        return [pad_impl(data, pads, self.mode, constant_value)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/07c494bf077e9e4a7898119f28a50585469ad4cd/onnx/backend/test/case/node/pad.py#L15-L41
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def pad_impl(data, raw_pads, mode, constant_values=0.0):  # type: ignore

    input_rank = data.ndim
    if input_rank * 2 != raw_pads.size:
        raise Exception("The number of elements in raw_pads should be 2 * data_rank")

    # re-order to np.pad accepted order ((x1_begin, x1_end), (x2_begin, x2_end), ...)
    pad_width = ()
    for i in range(int(raw_pads.size / 2)):
        pad_width += (((raw_pads[i], raw_pads[i + input_rank])),)  # type: ignore

    if mode == "constant":
        y = np.pad(
            data,
            pad_width=pad_width,
            mode=mode,
            constant_values=constant_values,
        )
        return y

    y = np.pad(
        data,
        pad_width=pad_width,
        mode=mode,
    )

    return y
