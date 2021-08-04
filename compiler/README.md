# onnion

## Installation

```
$ pip3 install onnion
```

## Usage

```
$ onnion ssd-10-post.onnx -o ssd_post_model.py
$ python
>>> from ssd_post_model import init_graph
>>> graph = init_graph()
>>> inputs = ... # List[np.array]
>>> outputs = graph.run(*inputs)
```

The order of the inputs and the outputs in the `run` method corresponds to the order of the inputs and the outputs in the onnx graph.

See also [tutorial](https://github.com/Idein/onnion/tree/master#tutorial).

## Development Guide

```
$ poetry install
```
