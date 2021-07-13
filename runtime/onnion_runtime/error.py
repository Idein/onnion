class RunError(Exception):
    def __init__(self, op, version):
        self.op = op
        self.version = version
