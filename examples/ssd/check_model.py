import onnxruntime
import numpy as np

from ssd_post import Graph0

if __name__ == "__main__":
    v0 = np.random.randn(1, 15130, 4).astype(np.float32)
    v1 = np.random.randn(1, 81, 15130).astype(np.float32)
    sess = onnxruntime.InferenceSession('ssd-10-post.onnx')
    expeced = sess.run(['bboxes', 'labels', 'scores'], {'Transpose_472': v0, 'Transpose_661': v1})

    y = Graph0().run(v0, v1)

    for a,b in zip(expeced, y):
        print("check")
        assert np.all(abs(a-b) < 1e-4)

    print("pass")
