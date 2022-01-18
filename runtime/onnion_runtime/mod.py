import numpy as np


class Mod:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.fmod = kwargs.get("fmod", 0)

    def run(self, x, y):
        if self.fmod == 0:
            return [np.mod(x, y)]
        else:
            return [np.fmod(x, y)]
