from importlib import import_module
import numpy as np
import onnx
import onnxruntime
import os
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 4
    target_dir = sys.argv[1]
    origin = sys.argv[2]
    module = os.path.splitext(sys.argv[3])[0]

    model = onnx.load(origin)
    output_names = [v.name for v in model.graph.output]

    inputs = dict()
    input_values = list()
    for i in model.graph.input:
        shape = list()
        for d in i.type.tensor_type.shape.dim:
            shape.append(d.dim_value)
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type]
        v = np.random.randn(*shape).astype(dtype)
        inputs[i.name] = v
        input_values.append(v)

    sess = onnxruntime.InferenceSession(origin)
    expeced = sess.run(output_names, inputs)

    onnion = import_module(f'.{module}', package=target_dir)
    outputs = onnion.Graph0().run(*input_values)

    for a,b in zip(expeced, outputs):
        print("check")
        assert np.all(abs(a-b) < 1e-4)

    print("pass")
