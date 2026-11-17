"""Đánh giá pixel chính xác cho hai bản đồ biên nhị phân cùng kích thước."""

import numpy as np


def _binary_mask(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("Ảnh biên phải là numpy.ndarray")
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Ảnh biên phải có dạng H x W không rỗng")
    if image.dtype == np.bool_:
        return image
    if image.dtype != np.uint8 or not np.all((image == 0) | (image == 255)):
        raise ValueError("Ảnh biên phải là bool hoặc uint8 chỉ chứa 0 và 255; hãy phân ngưỡng trước")
    return image != 0


def evaluate_edges(predicted, reference):
    """Trả TP/FP/FN/TN, Precision, Recall, F1, IoU.

    Ghép đúng từng pixel, không dung sai vị trí. Hai ảnh đều không có biên:
    cả bốn điểm bằng 1. Trường hợp khác: tỉ số có mẫu số 0 trả về 0.
    Reference phải được nêu rõ là OpenCV hay biên hình học chuẩn.
    """
    prediction = _binary_mask(predicted)
    truth = _binary_mask(reference)
    if prediction.shape != truth.shape:
        raise ValueError("Hai ảnh biên phải có cùng kích thước")

    tp = int(np.count_nonzero(prediction & truth))
    fp = int(np.count_nonzero(prediction & ~truth))
    fn = int(np.count_nonzero(~prediction & truth))
    tn = int(np.count_nonzero(~prediction & ~truth))
    if tp + fp + fn == 0:
        precision = recall = f1 = iou = 1.0
    else:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall, f1=f1, iou=iou)
