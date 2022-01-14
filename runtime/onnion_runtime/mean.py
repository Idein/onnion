from .error import RunError


class Mean:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.consumed_inputs = kwargs.get("consumed_inputs")

    def run(self, *xs):
        n = len(xs)
        if n < 1:
            raise RunError("Mean", self.version)
        else:
            acc = xs[0]
            for x in xs[1:]:
                acc = acc + x
            return [acc / n]
