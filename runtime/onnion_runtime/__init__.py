from .abs import Abs  # noqa: F401
from .acos import Acos  # noqa: F401
from .acosh import Acosh  # noqa: F401
from .add import Add  # noqa: F401
from .and_ import And  # noqa: F401
from .argmax import ArgMax  # noqa: F401
from .argmin import ArgMin  # noqa: F401
from .asin import Asin  # noqa: F401
from .asinh import Asinh  # noqa: F401
from .atan import Atan  # noqa: F401
from .atanh import Atanh  # noqa: F401
from .bitshift import BitShift  # noqa: F401
from .cast import Cast  # noqa: F401
from .ceil import Ceil  # noqa: F401
from .celu import Celu  # noqa: F401
from .clip import Clip  # noqa: F401
from .compress import Compress  # noqa: F401
from .concat import Concat  # noqa: F401
from .concatfromsequence import ConcatFromSequence  # noqa: F401
from .constant import Constant  # noqa: F401
from .constantofshape import ConstantOfShape  # noqa: F401
from .cos import Cos  # noqa: F401
from .cosh import Cosh  # noqa: F401
from .depthtospace import DepthToSpace  # noqa: F401
from .dequantizelinear import DequantizeLinear  # noqa: F401
from .det import Det  # noqa: F401
from .div import Div  # noqa: F401
from .dropout import Dropout  # noqa: F401
from .dynamicquantizelinear import DynamicQuantizeLinear  # noqa: F401
from .einsum import Einsum  # noqa: F401
from .elu import Elu  # noqa: F401
from .equal import Equal  # noqa: F401
from .erf import Erf  # noqa: F401
from .error import RunError  # noqa: F401
from .exp import Exp  # noqa: F401
from .expand import Expand  # noqa: F401
from .eyelike import EyeLike  # noqa: F401
from .flatten import Flatten  # noqa: F401
from .floor import Floor  # noqa: F401
from .gather import Gather  # noqa: F401
from .gatherelements import GatherElements  # noqa: F401
from .gathernd import GatherND  # noqa: F401
from .gemm import Gemm  # noqa: F401
from .globalaveragepool import GlobalAveragePool  # noqa: F401
from .globalmaxpool import GlobalMaxPool  # noqa: F401
from .greater import Greater  # noqa: F401
from .greaterorequal import GreaterOrEqual  # noqa: F401
from .hardmax import Hardmax  # noqa: F401
from .hardsigmoid import HardSigmoid  # noqa: F401
from .hardswish import HardSwish  # noqa: F401
from .identity import Identity  # noqa: F401
from .if_ import If  # noqa: F401
from .instancenormalization import InstanceNormalization  # noqa: F401
from .isinf import IsInf  # noqa: F401
from .isnan import IsNaN  # noqa: F401
from .leakyrelu import LeakyRelu  # noqa: F401
from .less import Less  # noqa: F401
from .lessorequal import LessOrEqual  # noqa: F401
from .log import Log  # noqa: F401
from .logsoftmax import LogSoftmax  # noqa: F401
from .loop import Loop  # noqa: F401
from .matmul import MatMul  # noqa: F401
from .matmulinteger import MatMulInteger  # noqa: F401
from .max import Max  # noqa: F401
from .mean import Mean  # noqa: F401
from .min import Min  # noqa: F401
from .mod import Mod  # noqa: F401
from .mul import Mul  # noqa: F401
from .neg import Neg  # noqa: F401
from .negativeloglikelihoodloss import NegativeLogLikelihoodLoss  # noqa: F401
from .nonmaxsuppression import NonMaxSuppression  # noqa: F401
from .nonzero import NonZero  # noqa: F401
from .not_ import Not  # noqa: F401
from .onehot import OneHot  # noqa: F401
from .or_ import Or  # noqa: F401
from .pad import Pad  # noqa: F401
from .pow import Pow  # noqa: F401
from .prelu import PRelu  # noqa: F401
from .randomnormal import RandomNormal  # noqa: F401
from .randomnormallike import RandomNormalLike  # noqa: F401
from .randomuniform import RandomUniform  # noqa: F401
from .randomuniformlike import RandomUniformLike  # noqa: F401
from .range import Range  # noqa: F401
from .reciprocal import Reciprocal  # noqa: F401
from .reducel1 import ReduceL1  # noqa: F401
from .reducel2 import ReduceL2  # noqa: F401
from .reducelogsum import ReduceLogSum  # noqa: F401
from .reducelogsumexp import ReduceLogSumExp  # noqa: F401
from .reducemax import ReduceMax  # noqa: F401
from .reducemean import ReduceMean  # noqa: F401
from .reducemin import ReduceMin  # noqa: F401
from .reduceprod import ReduceProd  # noqa: F401
from .reducesum import ReduceSum  # noqa: F401
from .reducesumsquare import ReduceSumSquare  # noqa: F401
from .relu import Relu  # noqa: F401
from .reshape import Reshape  # noqa: F401
from .round import Round  # noqa: F401
from .scatternd import ScatterND  # noqa: F401
from .shape import Shape  # noqa: F401
from .sigmoid import Sigmoid  # noqa: F401
from .slice import Slice  # noqa: F401
from .squeeze import Squeeze  # noqa: F401
from .sub import Sub  # noqa: F401
from .tile import Tile  # noqa: F401
from .topk import TopK  # noqa: F401
from .transpose import Transpose  # noqa: F401
from .unsqueeze import Unsqueeze  # noqa: F401
from .where import Where  # noqa: F401


def is_supported(op_name):
    if op_name == "RunError":
        return False
    else:
        return op_name in globals()


def warning(op_name, opset_version):
    if is_supported(op_name):
        op_class = globals()[op_name]
        if hasattr(op_class, "warning"):
            return op_class.warning(opset_version)
        else:
            return None
    else:
        return None
