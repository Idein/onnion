import numpy as np

from .error import RunError


class DepthToSpace:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.blocksize = kwargs["blocksize"]
        self.mode = kwargs.get("mode", "DCR")

    def run(self, x):
        assert len(x.shape) == 4
        n, c, h, w = x.shape

        if self.mode == "DCR":
            tmp = np.reshape(x, [n, self.blocksize, self.blocksize, c // (self.blocksize ** 2), h, w])
            tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
            y = np.reshape(tmp, [n, c // (self.blocksize ** 2), h * self.blocksize, w * self.blocksize])
            return [y]
        elif self.mode == "CRD":
            tmp = np.reshape(x, [n, c // (self.blocksize ** 2), self.blocksize, self.blocksize, h, w])
            tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
            y = np.reshape(tmp, [n, c // (self.blocksize ** 2), h * self.blocksize, w * self.blocksize])
            return [y]
        else:
            raise RunError("DepthToSpace", self.version)
