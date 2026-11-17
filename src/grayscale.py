import numpy as np

def rgb_to_grayscale(img_rgb):
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError("Ảnh phải là numpy.ndarray")

    if img_rgb.size == 0:
        raise ValueError("Ảnh không được rỗng")

    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Ảnh đầu vào phải là ảnh RGB có dạng H x W x 3")

    if img_rgb.dtype != np.uint8:
        raise ValueError("Ảnh phải có kiểu dữ liệu uint8")

    red = img_rgb[:, :, 0].astype(np.float32)
    green = img_rgb[:, :, 1].astype(np.float32)
    blue = img_rgb[:, :, 2].astype(np.float32)

    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    gray = np.rint(gray)
    gray = gray.astype(np.uint8)

    return gray