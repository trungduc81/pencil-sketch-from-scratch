import cv2
import numpy as np


def opencv_pencil_sketch(img_rgb, smoothing_kernel_size=5, smoothing_sigma=1.0, sketch_kernel_size=21, sketch_sigma=5.0):
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError("Ảnh phải là numpy.ndarray")

    if img_rgb.size == 0:
        raise ValueError("Ảnh không được rỗng")

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải là ảnh RGB có dạng H x W x 3")

    if img_rgb.dtype != np.uint8:
        raise ValueError("Ảnh phải có kiểu dữ liệu uint8")

    if not isinstance(smoothing_kernel_size, int) or not isinstance(sketch_kernel_size, int):
        raise TypeError("Kích thước kernel phải là số nguyên")

    if smoothing_kernel_size <= 0 or smoothing_kernel_size % 2 == 0:
        raise ValueError("Kernel làm mịn phải là số lẻ dương")

    if sketch_kernel_size <= 0 or sketch_kernel_size % 2 == 0:
        raise ValueError("Kernel tạo nét chì phải là số lẻ dương")

    if not isinstance(smoothing_sigma, (int, float)) or not isinstance(sketch_sigma, (int, float)):
        raise TypeError("Sigma phải là một số")

    if smoothing_sigma <= 0 or sketch_sigma <= 0:
        raise ValueError("Sigma phải lớn hơn 0")

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    smoothed = cv2.GaussianBlur(gray, (smoothing_kernel_size, smoothing_kernel_size), smoothing_sigma, borderType=cv2.BORDER_REFLECT_101)

    gradient_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    gradient_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    max_value = np.max(magnitude)

    if max_value == 0:
        edges = np.zeros(gray.shape, dtype=np.uint8)
    else:
        edges = magnitude / max_value * 255
        edges = np.rint(edges)
        edges = np.clip(edges, 0, 255)
        edges = edges.astype(np.uint8)

    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (sketch_kernel_size, sketch_kernel_size), sketch_sigma, borderType=cv2.BORDER_REFLECT_101)

    denominator = 255 - blurred.astype(np.float32)
    denominator = np.maximum(denominator, 1)
    sketch = cv2.divide(gray.astype(np.float32), denominator, scale=255)
    sketch = np.rint(sketch)
    sketch = np.clip(sketch, 0, 255)
    sketch = sketch.astype(np.uint8)

    dark_edges = cv2.bitwise_not(edges)
    result = cv2.min(sketch, dark_edges)

    return result
