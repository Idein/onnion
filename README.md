# onnion project

- compile onnx to python
- runtime depends only numpy

See [supported operators](https://github.com/Idein/onnion/tree/master/runtime#supported-operators).

## Tutorial
Extract the post-process of [Ultraface](https://github.com/onnx/models/tree/master/vision/body_analysis/ultraface) and run it.

Install tools:

```
$ pip3 install onnigiri onnion onnion-rt
```

Download the onnx file:

```
$ wget -O ultraface.onnx 'https://github.com/onnx/models/raw/master/vision/body_analysis/ultraface/models/version-RFB-640.onnx'
```

Extract the post-process:

```
$ onnigiri ultraface.onnx -o ultraface-post.onnx --from 460 --to boxes
```

Compile onnx to python:

```
$ onnion ultraface-post.onnx -o ultraface_post.py
```

Run and check:

```
$ python check_model.py
check
pass
```

<details>
<summary>check_model.py</summary>

```py
import onnxruntime
import numpy as np

from ultraface_post import init_graph

if __name__ == "__main__":
    x = np.random.randn(1,17640, 4).astype(np.float32)
    sess = onnxruntime.InferenceSession('ultraface-post.onnx')
    expeced = sess.run(['boxes'], {'460': x})

    graph = init_graph()
    y = graph.run(x)

    for a,b in zip(expeced, y):
        print("check")
        assert np.all(abs(a-b) < 1e-4)

    print("pass")
```
</details>

See also [examples](./examples).

## Development Guide
See each subdirectory:

- [onnion](https://github.com/Idein/onnion/tree/master/compiler#development-guide)
- [onnion-rt](https://github.com/Idein/onnion/tree/master/runtime#development-guide)

## Related project

- [onnigiri](https://github.com/Idein/onnigiri)
