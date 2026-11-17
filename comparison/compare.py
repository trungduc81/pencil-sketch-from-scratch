from pathlib import Path
import sys
import time

from comparison.metrics import compare_images
from opencv.opencv_reference import opencv_pencil_sketch
from src.image_io import load_image, save_image
from src.pipeline import pencil_sketch_pipeline


def main():
    input_path = sys.argv[1]
    img_rgb = load_image(input_path)

    start_time = time.perf_counter()
    self_result = pencil_sketch_pipeline(img_rgb)
    self_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    opencv_result = opencv_pencil_sketch(img_rgb)
    opencv_time = time.perf_counter() - start_time

    metrics = compare_images(self_result, opencv_result)
    image_name = Path(input_path).stem
    self_output = Path("results/from_scratch") / f"{image_name}_self.png"
    opencv_output = Path("results/opencv") / f"{image_name}_opencv.png"

    save_image(self_result, self_output)
    save_image(opencv_result, opencv_output)

    print(f"MSE: {metrics['mse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")

    if metrics["psnr"] == float("inf"):
        print("PSNR: inf")
    else:
        print(f"PSNR: {metrics['psnr']:.2f} dB")

    print(f"Thời gian tự code: {self_time:.4f} giây")
    print(f"Thời gian OpenCV: {opencv_time:.4f} giây")
    print(f"Kết quả tự code: {self_output}")
    print(f"Kết quả OpenCV: {opencv_output}")


if __name__ == "__main__":
    main()
