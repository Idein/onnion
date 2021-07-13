from .error import RunError


class Slice:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.starts = kwargs.get("starts")
        self.ends = kwargs.get("ends")
        self.axes = kwargs.get("axes")

    def run(self, x, starts=None, ends=None, axes=None, steps=None):
        if self.version >= 10:
            self.starts = starts
            self.ends = ends
            self.axes = axes

        if self.starts is None or self.ends is None:
            raise RunError("Slice", self.version)
        else:
            return [slice_ndarray(x, self.starts, self.ends, self.axes, steps)]


def slice_ndarray(x, starts, ends, axes, steps):
    if axes is None:
        ndim = len(starts)
        axes = list(range(ndim))

    idxs = [slice(0, size) for size in x.shape]

    if steps is None:
        idxs1 = [slice(s, e) for s, e in zip(starts, ends)]
    else:
        idxs1 = [slice(s, e, d) for s, e, d in zip(starts, ends, steps)]

    for i, a in enumerate(axes):
        idxs[a] = idxs1[i]

    return x[tuple(idxs)]
