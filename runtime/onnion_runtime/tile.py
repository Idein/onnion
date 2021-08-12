import numpy as np

from .error import RunError


class Tile:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x, repeats, axis=None):
        if self.version >= 6:
            return [np.tile(x, repeats)]
        else:
            raise RunError("Tile", self.version)
