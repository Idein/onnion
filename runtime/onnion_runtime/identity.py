class Identity:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version

    def run(self, x):
        if type(x) == list:
            return [[v.copy() for v in x]]
        else:
            return [x.copy()]
