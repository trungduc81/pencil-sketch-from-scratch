"""Sobel, Prewitt và Laplacian tự cài đặt bằng NumPy.

Ảnh đầu vào: uint8 H x W. Đạo hàm giữ float32 có dấu; bản đồ biên
để hiển thị/trộn tranh là uint8 0..255. convolve2d lật kernel nên
Gx, Gy giữ quy ước của code gốc: ngược dấu với cv2.Sobel.
"""

import numpy as np

from src.convolution import convolve2d


def _validate_gray(img):
    if not isinstance(img, np.ndarray):
        raise TypeError("Ảnh phải là numpy.ndarray")
    if img.ndim != 2 or img.size == 0:
        raise ValueError("Ảnh phải là ma trận H x W không rỗng")
    if img.dtype != np.uint8:
        raise ValueError("Ảnh đầu vào phải có kiểu uint8")


def normalize_edges(response):
    """Chuẩn hóa trị tuyệt đối của đáp ứng về uint8; ảnh hằng 0 -> 0."""
    strength = np.abs(response).astype(np.float32)
    maximum = np.max(strength)
    if maximum == 0:
        return np.zeros(strength.shape, dtype=np.uint8)
    return np.clip(np.rint(strength / maximum * 255), 0, 255).astype(np.uint8)


def gradient_magnitude_direction(gradient_x, gradient_y):
    """Độ lớn thô và hướng gradient (radian, [-pi, pi]).

    Hướng tại pixel có độ lớn 0 được quy ước bằng 0. Đây là hướng
    gradient (pháp tuyến đường biên), không phải hướng tiếp tuyến biên.
    """
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    direction = np.arctan2(gradient_y, gradient_x)
    direction = np.where(magnitude == 0, 0, direction).astype(np.float32)
    return magnitude, direction


def sobel_gradients(img):
    """Gx, Gy float32; giữ nguyên kernel và dấu của phiên bản ban đầu."""
    _validate_gray(img)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    return convolve2d(img, kernel_x), convolve2d(img, kernel_x.T)


def sobel_edge_detection(img):
    """Bản đồ biên Sobel uint8, tương thích pipeline đang có."""
    gx, gy = sobel_gradients(img)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return normalize_edges(magnitude)


def prewitt_gradients(img):
    """Gx, Gy Prewitt: các trọng số làm mịn đều bằng 1."""
    _validate_gray(img)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    return convolve2d(img, kernel_x), convolve2d(img, kernel_x.T)


def prewitt_edge_detection(img):
    gx, gy = prewitt_gradients(img)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    return normalize_edges(magnitude)


def laplacian_response(img):
    """Đạo hàm bậc hai có dấu; kernel 4 lân cận, tổng trọng số bằng 0."""
    _validate_gray(img)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    return convolve2d(img, kernel)


def laplacian_edge_detection(img):
    """Cường độ |Laplacian|, không thực hiện dò zero-crossing."""
    return normalize_edges(laplacian_response(img))
