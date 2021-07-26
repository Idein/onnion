import onnxruntime
import numpy as np

from ultraface_post import Graph0

if __name__ == "__main__":
    x = np.random.randn(1,17640, 4).astype(np.float32)
    sess = onnxruntime.InferenceSession('ultraface-post.onnx')
    expeced = sess.run(['boxes'], {'460': x})

    y = Graph0().run(x)

    for a,b in zip(expeced, y):
        print("check")
        assert np.all(abs(a-b) < 1e-4)

    print("pass")
