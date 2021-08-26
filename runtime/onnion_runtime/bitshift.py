from .error import RunError


class BitShift:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.direction = kwargs["direction"]

    def run(self, x, y):
        if self.direction == "RIGHT":
            return [x >> y]
        elif self.direction == "LEFT":
            return [x << y]
        else:
            raise RunError("BitShift", self.version)
