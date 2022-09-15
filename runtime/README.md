# onnion-rt

Note: This software includes [the work](https://github.com/onnx/onnx) that is distributed in the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0.html).

## Installation

```
$ pip3 install onnion-rt
```

## Usage
See [tutorial](https://github.com/Idein/onnion/tree/master#tutorial).

## Development Guide

```
$ poetry install
```

### How to support new operators

1. Add `onnion_runtime/xxx.py`
  - An onnx operator `Xxx` must correspond to a class `Xxx`.
  - A class `Xxx` must implement `__init__` and `run` methods.
  - The parameters of the `__init__` methods must be `self`, `opset_version`, and `kwargs`.
  - The attributes of the operator must be passed through the `kwargs` of the `__init__` method.
    - Get the required attributes by `kwargs['attr_name']`.
    - Get the optional attributes by `kwargs.get('attr_name', default_value)`.
  - The inputs of the operator must be passed through the arguments of the `run` method.
  - The `run` method must return the list of `np.array`.
2. Add `from .xxx import Xxx # noqa: F401` to `onnion_runtime/__init__.py`
3. Update "Supported Operators" in `README.md`
4. Add `tests/test_xxx.py`
5. Run tests `poetry run pytest -v`
6. Format and lint `poetry run pysen run format && poetry run pysen run lint`

## Supported Operators
This runtime supports only below operators.

- Abs
- Acos
- Acosh
- Add
  - must be from opsetversion >= 7
- And
  - must be from opsetversion >= 7
- ArgMax
- ArgMin
- Asin
- Asinh
- Atan
- Atanh
- BitShift
- Cast
  - must be from opsetversion >= 6
- Ceil
- Celu
- Clip
- Compress
- Concat
- ConcatFromSequence
- Constant
- ConstantOfShape
- Cos
- Cosh
- DepthToSpace
- DequantizeLinear
- Det
- Div
  - must be from opsetversion >= 7
- Dropout
- DynamicQuantizeLinear
- Einsum
- Elu
- Equal
  - must be from opsetversion >= 7
- Erf
- Exp
- Expand
- EyeLike
- Flatten
- Floor
- Gather
- GatherElements
- GahterND
- Gemm
  - must be from opsetversion >= 7
- GlobalAveragePool
- GlobalMaxPool
- Greater
  - must be from opsetversion >= 7
- GreaterOrEqual
- HardSigmoid
- HardSwish
- Hardmax
- Identity
- If
- InstanceNormalization
- IsInf
- IsNaN
- LeakyRelu
- Less
  - must be from opsetversion >= 7
- LessOrEqual
- Log
- LogSoftmax
- Loop
- MatMul
- MatMulInteger
- Max
- Mean
- Min
- Mod
- Mul
  - must be from opsetversion >= 7
- Neg
- NegativeLogLikelihoodLoss
- NonMaxSuppression
- NonZero
- Not
- OneHot
- Or
  - must be from opsetversion >= 7
- PRelu
- Pad
- Pow
  - must be from opsetversion >= 7
- RandomNormal
- RandomNormalLike
- RandomUniform
- RandomUniformLike
- Range
- Reciprocal
- ReduceL1
- ReduceL2
- ReduceLogSum
- ReduceLogSumExp
- ReduceMax
- ReduceMean
- ReduceMin
- ReduceProd
- ReduceSum
- ReduceSumSquare
- Relu
- Reshape
- Round
- ScatterND
- Shape
- Sigmoid
- Slice
- Split
  - argument `split` must be specified
- Squeeze
- Sub
  - must be from opsetversion >= 7
- Tile
  - must be from opsetversion >= 6
- TopK
- Transpose
- Unsqueeze
- Where
