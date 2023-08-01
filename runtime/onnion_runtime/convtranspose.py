from typing import Any, List, Optional

import numpy as np

from .error import RunError


# https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
class ConvTranspose:
    auto_pad: str
    group: int
    dilations: Optional[List[int]]
    strides: Optional[List[int]]
    kernel_shape: Optional[List[int]]
    output_shape: Optional[List[int]]
    output_padding: Optional[List[int]]
    pads: Optional[List[int]]

    def __init__(self, opset_version: int, **kwargs: Any):
        self.version = opset_version
        self.auto_pad = kwargs.get("auto_pad", "NOTSET")
        self.dilations = kwargs.get("dilations", None)
        self.group = kwargs.get("group", 1)
        self.kernel_shape = kwargs.get("kernel_shape", None)
        self.output_padding = kwargs.get("output_padding", None)
        self.output_shape = kwargs.get("output_shape", None)
        self.pads = kwargs.get("pads", None)
        self.strides = kwargs.get("strides", None)

    def run(self, x: np.ndarray, W: np.ndarray, b: Optional[np.ndarray] = None) -> List[np.ndarray]:
        """
        2D Convolution Transpose
        input shapes:
            x: [batch, in_ch, in_h, in_w]
            W: [in_ch, out_ch/group, kernel_h, kernel_w]
            b: [out_ch]
        output shape:
            [batch, out_ch, out_h, out_w]
        """

        # define parameters
        dim = len(x.shape) - 2
        group = self.group
        batch = x.shape[0]
        out_ch = W.shape[1]
        dilations = self.dilations or [1] * dim
        strides = self.strides or [1] * dim
        output_padding = self.output_padding or [0] * dim
        kernel_shape = self.kernel_shape or W.shape[2:]
        input_shape = x.shape[2:]
        pads = self.pads or [0] * (dim * 2)

        if dim != 2:
            raise RunError("ConvTranspose", self.version, "support 2d only")

        if group != 1:
            raise RunError("ConvTranspose", self.version, "support group=1 only")

        if self.auto_pad != "NOTSET":
            raise RunError("ConvTranspose", self.version, "support auto_pad=NOTSET only")
        # calculate pads and output_shape
        if self.output_shape is not None:
            output_shape = self.output_shape
            total_padding = [
                strides[i] * (input_shape[i] - 1)
                + output_padding[i]
                + ((kernel_shape[i] - 1) * dilations[i] + 1)
                - output_shape[i]
                for i in range(len(input_shape))
            ]
            for i in range(len(input_shape)):
                pads[i] = total_padding[i] - (total_padding[i] // 2)
                pads[i + dim] = total_padding[i] // 2
        else:
            output_shape = [
                strides[i] * (input_shape[i] - 1)
                + output_padding[i]
                + ((kernel_shape[i] - 1) * dilations[i] + 1)
                - pads[i]
                - pads[i + dim]
                for i in range(dim)
            ]

        # calculate output
        result = np.zeros([batch, out_ch, *output_shape], dtype=x.dtype)
        for och in range(out_ch):
            if b is not None:
                result[:, och, :, :] += b[och]
        for n in range(batch):
            for ih in range(input_shape[0]):
                for iw in range(input_shape[1]):
                    oh_min = max(strides[0] * ih - pads[0], 0)
                    oh_max = min(strides[0] * ih + kernel_shape[0] * dilations[0] - pads[0], output_shape[0])
                    ow_min = max(strides[1] * iw - pads[1], 0)
                    ow_max = min(strides[1] * iw + kernel_shape[1] * dilations[1] - pads[1], output_shape[1])
                    v = np.sum(x[n, :, ih, iw].reshape(-1, 1, 1, 1) * W, axis=0)
                    result[n, :, oh_min : oh_max : dilations[0], ow_min : ow_max : dilations[1]] += v

        return [result]
