from .error import RunError


class Add:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.axis = kwargs.get("axis")
        self.broadcast = kwargs.get("broadcast")
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, x, y):
        if self.version > 6:
            return [x + y]
        else:
            raise RunError("Add", self.version)
