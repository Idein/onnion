import numpy as np


class NegativeLogLikelihoodLoss:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.ignore_index = kwargs.get("ignore_index")
        self.reduction = kwargs.get("reduction", "mean")

    def run(self, x, target, weight=None):
        return [
            compute_negative_log_likelihood_loss(x, target, weight, reduction=self.reduction, ignore_index=self.ignore_index)
        ]


# The following code has been copied from
# https://github.com/onnx/onnx/blob/793ddeda9e2de8b71bc2d1e6d957c4e6d1274e5c/onnx/backend/test/case/node/negativeloglikelihoodloss.py#L15-L73 # noqa: B950
# Copyrights (c) ONNX Project Contributers
# License: Apache-2.0
def compute_negative_log_likelihood_loss(input, target, weight=None, reduction="mean", ignore_index=None):  # type: ignore
    input_shape = input.shape
    if len(input_shape) == 1:
        raise RuntimeError("Unsupported shape")

    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]

    # initialize the positional weights when required
    gather_weight = None
    if weight is not None:
        # setting mode='clip' to deal with ignore_index > C or < 0 cases.
        # when the target value is > C or < 0, it doesn't matter which value we are
        # taking in gather_weight, since it will be set to 0 in the following if-block
        # use np.int32 to make it compatible with x86 machines
        gather_weight = np.take(weight, np.array(target, dtype=np.int32), mode="clip")
        # set `ignore_index`'s loss weight to 0.
        # The loss tensor will be multiplied by this weight tensor,
        # so `ingore_index`'s loss value will be eliminated.
        if ignore_index is not None:
            gather_weight = np.where(target == ignore_index, 0, gather_weight).astype(dtype=np.float32)
    elif ignore_index is not None:
        gather_weight = np.where(target == ignore_index, 0, 1).astype(dtype=np.float32)

    # if input is 4-d and above, make it 3-d
    if len(input_shape) != 3:
        input = input.reshape((N, C, -1))
        target = target.reshape((N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = input.shape[2]
    neg_gather_element_input = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        for d in range(D):
            if target[i][d] != ignore_index:
                neg_gather_element_input[i][d] = -input[i][target[i][d]][d]

    loss = neg_gather_element_input

    # if the input was 4-d or above reshape to the right shape
    if len(input_shape) != 3:
        loss = loss.reshape(target_shape)

    # apply the weights when required
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = loss.sum() / gather_weight.sum()
            return loss

    if reduction == "mean":
        loss = np.mean(loss)
    elif reduction == "sum":
        loss = np.sum(loss)
    return loss
