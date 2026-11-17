import numpy as np
from src.smoothing import gaussian_filter

def invert_img(img):
    inverted = 255 - img
    return inverted

def color_dodge(gray, blurred):
    gray_float = gray.astype(np.float32)
    blurred_float = blurred.astype(np.float32)

    denominator = np.maximum(255 - blurred_float, 1)

    sketch = gray_float * 255 / denominator
    sketch = np.rint(sketch)
    sketch = np.clip(sketch, 0, 255)
    sketch = sketch.astype(np.uint8)

    return sketch


def combine_sketch_and_edges(sketch, edges):
    dark_edges = 255 - edges
    res = np.minimum(sketch, dark_edges)
    return res


def create_pencil_sketch(gray, edges, kernel_size=21, sigma=5.0):
    if not isinstance(gray, np.ndarray) or not isinstance(edges, np.ndarray):
        raise TypeError("Ảnh xám và ảnh biên phải là numpy.ndarray")

    if gray.size == 0 or edges.size == 0:
        raise ValueError("Ảnh không được rỗng")

    if gray.ndim != 2 or edges.ndim != 2:
        raise ValueError("Ảnh xám và ảnh biên phải có dạng H x W")

    if gray.shape != edges.shape:
        raise ValueError("Ảnh xám và ảnh biên phải có cùng kích thước")

    if gray.dtype != np.uint8 or edges.dtype != np.uint8:
        raise ValueError("Ảnh xám và ảnh biên phải có kiểu dữ liệu uint8")

    inverted = invert_img(gray)
    blurred = gaussian_filter(inverted, kernel_size, sigma)
    sketch = color_dodge(gray, blurred)
    res = combine_sketch_and_edges(sketch, edges)

    return res
