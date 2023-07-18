# onnion

## Installation
From [PyPI](https://pypi.org/project/onnion/):

```
$ pip3 install onnion
```

From [Dockerhub](https://hub.docker.com/repository/docker/idein/onnion):

```
docker pull idein/onnion:20230718
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

With docker:

```
$ docker run --rm -it -u $UID:$GID -v $(pwd):/work idein/onnion:20230718 ssd-10-post.onnx -o ssd_post_model.py
```

The order of the inputs and the outputs in the `run` method corresponds to the order of the inputs and the outputs in the onnx graph.

See also [tutorial](https://github.com/Idein/onnion/tree/master#tutorial).

## Development Guide

```
$ poetry install
```
