import numpy as np

from src.convolution import convolve2d

def gaussian_kernel(kernel_size=5, sigma=1.0):
    if not isinstance(kernel_size, int):
        raise TypeError("Kích thước kernel phải là số nguyên")

    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kích thước kernel phải là số lẻ dương")

    if not isinstance(sigma, (int, float)):
        raise TypeError("Sigma phải là một số")

    if sigma <= 0:
        raise ValueError("Sigma phải lớn hơn 0")

    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float32)
    x, y = np.meshgrid(coordinates, coordinates)

    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    return kernel.astype(np.float32)


def gaussian_filter(img, kernel_size=5, sigma=1.0):
    kernel = gaussian_kernel(kernel_size, sigma)
    res = convolve2d(img, kernel)

    res = np.rint(res)
    res = np.clip(res, 0, 255)
    res = res.astype(np.uint8)

    return res
