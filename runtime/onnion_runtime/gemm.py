import numpy as np

from .error import RunError


class Gemm:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.alpha = kwargs.get("alpha", 1.0)
        self.beta = kwargs.get("beta", 1.0)
        self.transA = kwargs.get("transA", 0)
        self.transB = kwargs.get("transB", 0)
        self.broadcast = kwargs.get("broadcast", 0)

    def run(self, a, b, c=None):
        if self.version >= 7:
            return [
                gemm_reference_implementation(a, b, c, alpha=self.alpha, beta=self.beta, transA=self.transA, transB=self.transB)
            ]
        else:
            raise RunError("Gemm", self.version)


# The following code has been copied from
# https://github.com/onnx/onnx/blob/a5e7ee51176bf78a60c118758174e13d85a87b46/onnx/backend/test/case/node/gemm.py#L16-L24
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
# NOTE: Remove type comment
def gemm_reference_implementation(A, B, C=None, alpha=1.0, beta=1.0, transA=0, transB=0):
    A = A if transA == 0 else A.T
    B = B if transB == 0 else B.T
    C = C if C is not None else np.array(0)

    Y = alpha * np.dot(A, B) + beta * C

    return Y
