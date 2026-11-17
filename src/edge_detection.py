import numpy as np
from src.convolution import convolve2d

def sobel_gradients(img):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    gradient_x = convolve2d(img, sobel_x)
    gradient_y = convolve2d(img, sobel_y)

    return gradient_x, gradient_y

def sobel_edge_detection(img):
    gradient_x, gradient_y = sobel_gradients(img)

    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    max_value = np.max(magnitude)

    if max_value == 0:
        return np.zeros(img.shape, dtype=np.uint8)

    magnitude = magnitude / max_value * 255
    magnitude = np.rint(magnitude)
    magnitude = np.clip(magnitude, 0, 255)
    magnitude = magnitude.astype(np.uint8)

    return magnitude
