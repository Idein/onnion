# onnion-rt

Note: This software includes [the work](https://github.com/onnx/onnx) that is distributed in the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).

## Installation

```
$ pip3 install onnion-rt
```

## Development Guide

```
$ poetry install
```

### How to support new operators

1. Add `onnion_runtime/xxx.py`
2. Add `from .xxx import Xxx # noqa: F401` to `onnion_runtime/__init__.py`
3. Update "Supported Operators" in `README.md`
4. Add `tests/test_xxx.py`
5. Run tests `poetry run pytest -v`
6. Format and lint `poetry run pysen run format && poetry run pysen run lint`

## Supported Operators
This runtime supports only below operators.

- Concat
- Exp
