import numpy as np
from onnion_runtime import NegativeLogLikelihoodLoss

from .utils import check


def test_negativeloglikelihoodloss_00():
    opset_version = 13
    attrs = {"reduction": "none"}

    n, c = 3, 5
    x = np.random.rand(n, c).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n,)).astype(np.int64)
    inputs = [x, target]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_01():
    opset_version = 13
    attrs = {"reduction": "mean"}

    n, c, d1 = 3, 5, 2
    x = np.random.rand(n, c, d1).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1)).astype(np.int64)
    inputs = [x, target]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_02():
    opset_version = 13
    attrs = {"reduction": "mean", "ignore_index": 1}

    n, c, d1 = 3, 5, 2
    x = np.random.rand(n, c, d1).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1)).astype(np.int64)
    target[0][0] = 1
    inputs = [x, target]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_03():
    opset_version = 13
    attrs = {"reduction": "mean", "ignore_index": -1}

    n, c, d1 = 3, 5, 6
    x = np.random.rand(n, c, d1).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1)).astype(np.int64)
    target[0][0] = -1
    weight = np.random.rand(c).astype(np.float32)
    inputs = [x, target, weight]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_04():
    opset_version = 13
    attrs = {"reduction": "mean"}

    n, c, d1 = 3, 5, 6
    x = np.random.rand(n, c, d1).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1)).astype(np.int64)
    weight = np.random.rand(c).astype(np.float32)
    inputs = [x, target, weight]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_05():
    opset_version = 13
    attrs = {"reduction": "mean", "ignore_index": 1}

    n, c, d1 = 3, 5, 6
    x = np.random.rand(n, c, d1).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1)).astype(np.int64)
    target[0][0] = 1
    weight = np.random.rand(c).astype(np.float32)
    inputs = [x, target, weight]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_06():
    opset_version = 13
    attrs = {"reduction": "none"}

    n, c, d1, d2 = 3, 5, 6, 6
    x = np.random.rand(n, c, d1, d2).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2)).astype(np.int64)
    inputs = [x, target]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_07():
    opset_version = 13
    attrs = {"reduction": "mean", "ignore_index": 1}

    n, c, d1, d2 = 3, 5, 6, 6
    x = np.random.rand(n, c, d1, d2).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2)).astype(np.int64)
    target[0][0][0] = 1
    inputs = [x, target]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_08():
    opset_version = 13
    attrs = {"reduction": "mean"}

    n, c, d1, d2 = 3, 5, 6, 6
    x = np.random.rand(n, c, d1, d2).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2)).astype(np.int64)
    inputs = [x, target]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_09():
    opset_version = 13
    attrs = {"reduction": "sum"}

    n, c, d1, d2 = 3, 5, 6, 6
    x = np.random.rand(n, c, d1, d2).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2)).astype(np.int64)
    inputs = [x, target]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_10():
    opset_version = 13
    attrs = {"reduction": "none"}

    n, c, d1, d2 = 3, 5, 6, 6
    x = np.random.rand(n, c, d1, d2).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2)).astype(np.int64)
    weight = np.random.rand(c).astype(np.float32)
    inputs = [x, target, weight]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_11():
    opset_version = 13
    attrs = {"reduction": "mean"}

    n, c, d1, d2 = 3, 5, 6, 6
    x = np.random.rand(n, c, d1, d2).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2)).astype(np.int64)
    weight = np.random.rand(c).astype(np.float32)
    inputs = [x, target, weight]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_12():
    opset_version = 13
    attrs = {"reduction": "sum"}

    n, c, d1, d2 = 3, 5, 6, 6
    x = np.random.rand(n, c, d1, d2).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2)).astype(np.int64)
    weight = np.random.rand(c).astype(np.float32)
    inputs = [x, target, weight]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_13():
    opset_version = 13
    attrs = {"reduction": "sum", "ignore_index": 0}

    n, c, d1, d2 = 3, 5, 6, 6
    x = np.random.rand(n, c, d1, d2).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2)).astype(np.int64)
    weight = np.random.rand(c).astype(np.float32)
    target[0][0][0] = 0
    inputs = [x, target, weight]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)


def test_negativeloglikelihoodloss_14():
    opset_version = 13
    attrs = {"reduction": "none", "ignore_index": -5}

    n, c, d1, d2, d3 = 3, 5, 6, 6, 5
    x = np.random.rand(n, c, d1, d2, d3).astype(np.float32)
    target = np.random.randint(0, high=c, size=(n, d1, d2, d3)).astype(np.int64)
    target[0][0][0][0] = -5
    inputs = [x, target]

    check(NegativeLogLikelihoodLoss, opset_version, attrs, inputs)
