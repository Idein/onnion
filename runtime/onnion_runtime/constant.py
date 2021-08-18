import numpy as np

from .error import RunError


class Constant:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.sparse_value = kwargs.get("sparse_value")
        self.value = kwargs.get("value")
        self.value_float = kwargs.get("value_float")
        self.value_floats = kwargs.get("value_floats")
        self.value_int = kwargs.get("value_int")
        self.value_ints = kwargs.get("value_ints")
        self.value_string = kwargs.get("value_string")
        self.value_strings = kwargs.get("value_strings")

    def run(self):
        if self.version < 11:
            if self.value is not None:
                return [self.value]
            else:
                raise RunError("Constant", self.version)
        elif self.version == 11:
            if self.value is not None:
                return [self.value]
            elif self.sparse_value is not None:
                return [self.sparse_value]
            else:
                raise RunError("Constant", self.version)
        else:
            if self.value is not None:
                return [self.value]
            elif self.sparse_value is not None:
                return [self.sparse_value]
            elif self.value_float is not None:
                return [np.array(self.value_float).astype(np.float32)]
            elif self.value_floats is not None:
                return [np.array(self.value_floats).astype(np.float32)]
            elif self.value_int is not None:
                return [np.array(self.value_int).astype(np.int64)]
            elif self.value_ints is not None:
                return [np.array(self.value_ints).astype(np.int64)]
            elif self.value_string is not None:
                return [np.array(self.value_string)]
            elif self.value_strings is not None:
                return [np.array(self.value_strings)]
            else:
                raise RunError("Constant", self.version)
