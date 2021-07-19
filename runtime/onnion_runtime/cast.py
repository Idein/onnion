from .error import RunError
from .utils import tensor_type_to_dtype


class Cast:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.to = kwargs["to"]

    def run(self, x):
        if self.version < 6:
            raise RunError("Cast", self.version)
        else:
            t = tensor_type_to_dtype(self.to)
            return [x.astype(t)]
