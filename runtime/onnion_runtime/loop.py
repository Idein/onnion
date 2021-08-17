import numpy as np


class Loop:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.body = kwargs["body"]

    def run(self, M, cond, *v_initial):
        if M == np.array(""):
            max_trip_count = np.array(np.inf, dtype=np.int64)
        else:
            max_trip_count = M.copy()

        if cond == np.array(""):
            loop_cond = np.array(True, dtype=np.bool)
        else:
            loop_cond = cond.copy()

        iter_count = np.array(0, dtype=np.int64)
        step_inputs = [v for v in v_initial]
        n = len(v_initial)

        while iter_count < max_trip_count and loop_cond:
            loop_cond, *step_outputs = self.body.run(iter_count, loop_cond, *step_inputs)
            step_inputs = [v for v in step_outputs[:n]]
            if iter_count == 0:
                scan_out = [(v,) for v in step_outputs[n:]]
            else:
                scan_out = [acc + (new,) for (acc, new) in zip(scan_out, step_outputs[n:])]
            iter_count += 1

        return step_outputs[:n] + [np.stack(vs) for vs in scan_out]
