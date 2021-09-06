import numpy as np


class NonMaxSuppression:
    def __init__(self, opset_version, **kwargs):
        self.version = opset_version
        self.center_point_box = kwargs.get("center_point_box", 0)

    # Ref: https://github.com/microsoft/onnxruntime/blob/55b26b69513d3972e7c0a70f0d54f3fbb8adc054/onnxruntime/core/providers/cpu/object_detection/non_max_suppression.cc#L117  # noqa: B950
    def run(self, boxes, scores, max_output_boxes_per_class=None, iou_threshold=None, score_threshold=None):

        if max_output_boxes_per_class is None:
            max_output_boxes_per_class = np.array([0], dtype=np.int64)
        if iou_threshold is None:
            iou_threshold = np.array([0], dtype=np.float32)
        if score_threshold is None:
            score_threshold = np.array([0], dtype=np.float32)

        num_batches, num_classes, _ = scores.shape
        selected_indices = list()

        for batch in range(num_batches):
            boxes_per_batch = boxes[batch, :, :]
            for cls in range(num_classes):
                scores_per_batch_per_class = scores[batch, cls, :]
                indices = np.argsort(-scores_per_batch_per_class, kind="stable")  # to pass test_nonmaxsuppression_02

                selected_indices_per_class = list()
                while len(indices) > 0 and len(selected_indices_per_class) < max_output_boxes_per_class:
                    idx0 = indices[0]
                    indices = indices[1:]

                    selected = scores_per_batch_per_class[idx0] > score_threshold
                    for idx in selected_indices_per_class:
                        if suppress_by_iou(boxes_per_batch, idx0, idx, self.center_point_box, iou_threshold):
                            selected = False
                            break

                    if selected:
                        selected_indices_per_class.append(idx0)
                        selected_indices.append([batch, cls, idx0])

        return [np.array(selected_indices).astype(np.int64)]


# Ref: https://github.com/microsoft/onnxruntime/blob/55b26b69513d3972e7c0a70f0d54f3fbb8adc054/onnxruntime/core/providers/cpu/object_detection/non_max_suppression_helper.h#L63 # noqa: B950
def suppress_by_iou(boxes, idx0, idx1, center_point_box, iou_threshold):
    box0 = boxes[idx0, :]
    box1 = boxes[idx1, :]

    if center_point_box == 0:
        x0_min = min(box0[1], box0[3])
        x0_max = max(box0[1], box0[3])
        x1_min = min(box1[1], box1[3])
        x1_max = max(box1[1], box1[3])

        intersection_x_min = max(x0_min, x1_min)
        intersection_x_max = min(x0_max, x1_max)
        if intersection_x_max <= intersection_x_min:
            return False

        y0_min = min(box0[0], box0[2])
        y0_max = max(box0[0], box0[2])
        y1_min = min(box1[0], box1[2])
        y1_max = max(box1[0], box1[2])

        intersection_y_min = max(y0_min, y1_min)
        intersection_y_max = min(y0_max, y1_max)
        if intersection_y_max <= intersection_y_min:
            return False

    else:
        w0 = box0[2]
        h0 = box0[3]
        w1 = box1[2]
        h1 = box1[3]

        x0_min = box0[0] - w0 / 2.0
        x0_max = box0[0] + w0 / 2.0
        x1_min = box1[0] - w1 / 2.0
        x1_max = box1[0] + w1 / 2.0

        intersection_x_min = max(x0_min, x1_min)
        intersection_x_max = min(x0_max, x1_max)
        if intersection_x_max <= intersection_x_min:
            return False

        y0_min = box0[1] - h0 / 2.0
        y0_max = box0[1] + h0 / 2.0
        y1_min = box1[1] - h1 / 2.0
        y1_max = box1[1] + h1 / 2.0

        intersection_y_min = max(y0_min, y1_min)
        intersection_y_max = min(y0_max, y1_max)
        if intersection_y_max <= intersection_y_min:
            return False

    intersection_area = (intersection_x_max - intersection_x_min) * (intersection_y_max - intersection_y_min)
    if intersection_area <= 0.0:
        return False

    area0 = (x0_max - x0_min) * (y0_max - y0_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    union_area = area0 + area1 - intersection_area
    if area0 <= 0.0 or area1 <= 0.0 or union_area <= 0.0:
        return False

    iou = intersection_area / union_area

    return iou > iou_threshold
