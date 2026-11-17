import numpy as np

def validate_images(img1, img2):
    if not isinstance(img1, np.ndarray) or not isinstance(img2, np.ndarray):
        raise TypeError("Hai ảnh phải là numpy.ndarray")

    if img1.size == 0 or img2.size == 0:
        raise ValueError("Ảnh không được rỗng")

    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Hai ảnh phải là ảnh xám có dạng H x W")

    if img1.shape != img2.shape:
        raise ValueError("Hai ảnh phải có cùng kích thước")

    if img1.dtype != np.uint8 or img2.dtype != np.uint8:
        raise ValueError("Hai ảnh phải có kiểu dữ liệu uint8")


def mean_squared_error(img1, img2):
    validate_images(img1, img2)

    difference = img1.astype(np.float32) - img2.astype(np.float32)
    mse = np.mean(difference ** 2)

    return float(mse)


def mean_absolute_error(img1, img2):
    validate_images(img1, img2)

    difference = img1.astype(np.float32) - img2.astype(np.float32)
    mae = np.mean(np.abs(difference))

    return float(mae)


def peak_signal_to_noise_ratio(img1, img2):
    mse = mean_squared_error(img1, img2)

    if mse == 0:
        return float("inf")

    psnr = 10 * np.log10((255 ** 2) / mse)
    return float(psnr)


def compare_images(img1, img2):
    mse = mean_squared_error(img1, img2)
    mae = mean_absolute_error(img1, img2)

    if mse == 0:
        psnr = float("inf")
    else:
        psnr = float(10 * np.log10((255 ** 2) / mse))

    return {"mse": mse, "mae": mae, "psnr": psnr}
