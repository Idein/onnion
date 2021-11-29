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

    def warning(opset_version):
        return """Cast may not work with raspi as you think:

x86>>> np.array([-2], dtype=np.float32).astype(np.uint32)
array([4294967294], dtype=uint32)
arm32>>> np.array([-2], dtype=np.float32).astype(np.uint32)
array([0], dtype=uint32)"""
