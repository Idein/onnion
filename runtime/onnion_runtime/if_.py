class If:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.then_branch = kwargs["then_branch"]
        self.else_branch = kwargs["else_branch"]

    def run(self, cond):
        if cond:
            return self.then_branch.run()
        else:
            return self.else_branch.run()
