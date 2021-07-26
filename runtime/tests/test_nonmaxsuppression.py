import numpy as np
from onnion_runtime import NonMaxSuppression

from .utils import check


def test_nonmaxsuppression_00():
    opset_version = 11
    center_point_box = 1

    boxes = np.array(
        [
            [
                [0.5, 0.5, 1.0, 1.0],
                [0.5, 0.6, 1.0, 1.0],
                [0.5, 0.4, 1.0, 1.0],
                [0.5, 10.5, 1.0, 1.0],
                [0.5, 10.6, 1.0, 1.0],
                [0.5, 100.5, 1.0, 1.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version, center_point_box=center_point_box).run(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
    )
    check("NonMaxSuppression", {"center_point_box": center_point_box}, inputs, outputs, opset_version)


def test_nonmaxsuppression_01():
    opset_version = 11

    boxes = np.array(
        [
            [
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, 0.9, 1.0, -0.1],
                [0.0, 10.0, 1.0, 11.0],
                [1.0, 10.1, 0.0, 11.1],
                [1.0, 101.0, 0.0, 100.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version).run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    check("NonMaxSuppression", dict(), inputs, outputs, opset_version)


def test_nonmaxsuppression_02():
    opset_version = 11

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version).run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    check("NonMaxSuppression", dict(), inputs, outputs, opset_version)


def test_nonmaxsuppression_03():
    opset_version = 11

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version).run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    check("NonMaxSuppression", dict(), inputs, outputs, opset_version)


def test_nonmaxsuppression_04():
    opset_version = 11

    boxes = np.array([[[0.0, 0.0, 1.0, 1.0]]]).astype(np.float32)
    scores = np.array([[[0.9]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version).run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    check("NonMaxSuppression", dict(), inputs, outputs, opset_version)


def test_nonmaxsuppression_05():
    opset_version = 11

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version).run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    check("NonMaxSuppression", dict(), inputs, outputs, opset_version)


def test_nonmaxsuppression_06():
    opset_version = 11

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([3]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.4]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version).run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    check("NonMaxSuppression", dict(), inputs, outputs, opset_version)


def test_nonmaxsuppression_07():
    opset_version = 11

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ],
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ],
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]], [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version).run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    check("NonMaxSuppression", dict(), inputs, outputs, opset_version)


def test_nonmaxsuppression_08():
    opset_version = 11

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3], [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = np.array([2]).astype(np.int64)
    iou_threshold = np.array([0.5]).astype(np.float32)
    score_threshold = np.array([0.0]).astype(np.float32)

    inputs = [boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold]
    outputs = NonMaxSuppression(opset_version).run(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)
    check("NonMaxSuppression", dict(), inputs, outputs, opset_version)
