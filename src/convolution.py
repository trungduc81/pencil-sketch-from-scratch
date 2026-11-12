import numpy as np

def pad_image(img, pad_height, pad_width):
    if not isinstance(pad_height, int) or not isinstance(pad_width, int):
        raise TypeError("Kích thước padding phải là số nguyên")

    if pad_height < 0 or pad_width < 0:
        raise ValueError("Kích thước padding không được âm")

    padded_img = np.pad(img, ((pad_height, pad_height),(pad_width, pad_width)) ,mode="reflect")
    return padded_img


def convolve2d(img, kernel):
    if not isinstance(img, np.ndarray):
        raise TypeError("Ảnh phải là numpy.ndarray")

    if img.size == 0:
        raise ValueError("Ảnh không được rỗng")

    if img.ndim != 2:
        raise ValueError("Ảnh phải có dạng H x W")

    if not isinstance(kernel, np.ndarray):
        raise TypeError("Kernel phải là numpy.ndarray")

    if kernel.size == 0:
        raise ValueError("Kernel không được rỗng")

    if kernel.ndim != 2:
        raise ValueError("Kernel phải là ma trận hai chiều")

    kernel_h, kernel_w = kernel.shape

    if kernel_h % 2 == 0 or kernel_w % 2 == 0:
        raise ValueError("Kích thước kernel phải là số lẻ")

    img_float = img.astype(np.float32)
    kernel_float = kernel.astype(np.float32)

    pad_height = kernel_h // 2
    pad_width = kernel_w // 2

    padded_img = pad_image(img_float, pad_height,pad_width)
    flipped_kernel = kernel_float[::-1, ::-1]
    h, w = img.shape

    result = np.zeros((h,w), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i + kernel_h,j:j + kernel_w]
            result[i,j] = np.sum(region * flipped_kernel)

    return result