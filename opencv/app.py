import time
import tkinter as tk

from opencv.opencv_reference import opencv_pencil_sketch
from src.app import PencilSketchApp


class OpenCVPencilSketchApp(PencilSketchApp):
    def __init__(self, root):
        super().__init__(root)
        self.root.title("Chuyển ảnh thành tranh chì - OpenCV")

    def process_image(self, smoothing_kernel_size, smoothing_sigma, sketch_kernel_size, sketch_sigma):
        start_time = time.perf_counter()

        try:
            sketch = opencv_pencil_sketch(self.img_rgb, smoothing_kernel_size, smoothing_sigma, sketch_kernel_size, sketch_sigma)
        except Exception as error:
            self.root.after(0, lambda message=str(error): self.processing_failed(message))
            return

        elapsed_time = time.perf_counter() - start_time
        self.root.after(0, lambda: self.processing_finished(sketch, elapsed_time))


def main():
    root = tk.Tk()
    OpenCVPencilSketchApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
