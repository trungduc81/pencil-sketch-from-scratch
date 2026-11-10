from pathlib import Path

import cv2
import numpy as np

valid_ext = ('.jpg','.jpeg','.png')

def load_image(input_path):
    path = Path(input_path)

    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {input_path}")

    if path.suffix.lower() not in valid_ext:
        raise ValueError(f"Định dạng ảnh không hỗ trợ (chỉ nhận {valid_ext}): {path.suffix}")

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(f"Không thể đọc ảnh hoặc file bị hỏng: {input_path}")

    img_rgb = img_bgr[:,:,::-1].copy()
    return img_rgb

def save_image(image, output_path):
    if not isinstance(image, np.ndarray):
        raise TypeError("Ảnh phải là numpy.ndarray")

    if image.size == 0:
        raise ValueError("Ảnh không được rỗng")

    if image.ndim != 2:
        raise ValueError("Tranh chì phải là ảnh xám có dạng H x W")

    if image.dtype != np.uint8:
        raise ValueError("Ảnh phải có kiểu dữ liệu uint8")

    path = Path(output_path)

    if path.suffix.lower() not in valid_ext:
        raise ValueError(f"Định dạng ảnh không hỗ trợ (chỉ nhận {valid_ext}): {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)

    if not success:
        raise IOError(f"Không thể lưu ảnh: {output_path}")
