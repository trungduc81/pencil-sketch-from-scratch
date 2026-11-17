"""Phân ngưỡng cố định cho bản đồ biên đã chuẩn hóa."""

import numpy as np


def threshold_edges(edges, threshold=60):
    """Pixel > threshold -> 255, còn lại -> 0 (giống THRESH_BINARY).

    Dùng dấu > để ảnh không có biên vẫn toàn 0 khi threshold = 0.
    Ngưỡng chỉ so sánh trong từng ảnh vì mỗi bản đồ được chuẩn hóa riêng.
    """
    if not isinstance(edges, np.ndarray):
        raise TypeError("Bản đồ biên phải là numpy.ndarray")
    if edges.ndim != 2 or edges.size == 0 or edges.dtype != np.uint8:
        raise ValueError("Bản đồ biên phải là uint8 H x W không rỗng")
    if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, (int, float, np.integer, np.floating)):
        raise TypeError("Ngưỡng phải là một số")
    if not np.isfinite(threshold) or not 0 <= threshold <= 255:
        raise ValueError("Ngưỡng phải nằm trong [0, 255]")
    return (edges > threshold).astype(np.uint8) * 255
