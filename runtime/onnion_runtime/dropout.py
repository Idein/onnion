import numpy as np


class Dropout:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.seed = kwargs.get("seed")
        self.consumed_inputs = kwargs.get("consumed_inputs")
        self.is_test = kwargs.get("is_test", 0)
        self.ratio = kwargs.get("ratio", 0.5)

    def run(self, data, ratio=None, training_mode=None):
        if self.version >= 12:
            if ratio is not None:
                self.ratio = ratio
            if training_mode is not None:
                return list(dropout(data, self.ratio, self.seed, training_mode.item(), return_mask=True))
            else:
                return list(dropout(data, self.ratio, self.seed, return_mask=True))
        else:
            return [data, np.zeros(data.shape, dtype=bool)]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/91bf3f52c96919af8de4d90c4e70316a47112178/onnx/backend/test/case/node/dropout.py#L17-L30
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def dropout(X, drop_probability=0.5, seed=0, training_mode=False, return_mask=False):  # type: ignore
    if drop_probability == 0 or training_mode is False:
        if return_mask is True:
            return X, np.ones(X.shape, dtype=bool)
        else:
            return X

    np.random.seed(seed)
    mask = np.random.uniform(0, 1.0, X.shape) >= drop_probability
    scale = 1 / (1 - drop_probability)
    if return_mask is True:
        return mask * X * scale, mask.astype(bool)
    else:
        return mask * X * scale
