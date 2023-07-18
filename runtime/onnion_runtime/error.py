from typing import Optional


class RunError(Exception):
    def __init__(self, op: str, version: int, reason: Optional[str] = None):
        self.op = op
        self.version = version
        self.reason = reason
