import numpy as np

class BatchNorm:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.gamma = kwargs.get("gamma")
        self.beta = kwargs.get("beta")
        self.eps = kwargs.get("eps")

    def run(self, xs):
        mean = np.average(xs)
        var  = np.var(xs)
        # normalize
        xhat = (xs - mean) / np.sqrt(var + self.eps)
        # scale and shift
        return [self.gamma * xhat + self.beta]

