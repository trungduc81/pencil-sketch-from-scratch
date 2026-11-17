from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.image_io import load_image, save_image
from src.pipeline import pencil_sketch_pipeline

DETAIL_PRESETS = {
    "Nhiều chi tiết": (3, 0.8),
    "Cân bằng": (5, 1.0),
    "Mịn, ít nhiễu": (7, 1.5)
}

PENCIL_PRESETS = {
    "Sắc nét": (11, 3.0),
    "Tự nhiên": (21, 5.0),
    "Mềm mại": (31, 7.0)
}

class PencilSketchApp:
    def __init__(self, root):
        self.root = root
        self.img_rgb = None
        self.sketch = None
        self.original_photo = None
        self.sketch_photo = None

        self.root.title("Chuyển ảnh thành tranh chì")
        self.root.geometry("1180x760")
        self.root.minsize(900, 650)

        self.detail_level = tk.StringVar(value="Cân bằng")
        self.pencil_style = tk.StringVar(value="Tự nhiên")
        self.status = tk.StringVar(value="Chọn một ảnh để bắt đầu")

        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("ImageTitle.TLabel", font=("Segoe UI", 12, "bold"))

        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Phần mềm chuyển ảnh thành tranh chì", style="Title.TLabel").pack(pady=(0, 14))

        controls = ttk.LabelFrame(container, text="Điều khiển", padding=12)
        controls.pack(fill="x", pady=(0, 14))

        self.open_button = ttk.Button(controls, text="Chọn ảnh", command=self.select_image)
        self.open_button.grid(row=0, column=0, rowspan=3, padx=(0, 24))

        ttk.Label(controls, text="Mức giữ chi tiết").grid(row=0, column=1, sticky="w")
        detail_box = ttk.Combobox(controls, textvariable=self.detail_level, values=list(DETAIL_PRESETS), state="readonly", width=18)
        detail_box.grid(row=1, column=1, padx=(0, 24), sticky="w")

        ttk.Label(controls, text="Kiểu nét chì").grid(row=0, column=2, sticky="w")
        pencil_box = ttk.Combobox(controls, textvariable=self.pencil_style, values=list(PENCIL_PRESETS), state="readonly", width=18)
        pencil_box.grid(row=1, column=2, padx=(0, 24), sticky="w")

        self.process_button = ttk.Button(controls, text="Tạo tranh chì", command=self.start_processing, state="disabled")
        self.process_button.grid(row=0, column=3, rowspan=3, padx=(0, 10))

        self.save_button = ttk.Button(controls, text="Lưu kết quả", command=self.save_result, state="disabled")
        self.save_button.grid(row=0, column=4, rowspan=3)

        controls.columnconfigure(5, weight=1)
        ttk.Label(controls, text="Có thể đổi lựa chọn và nhấn Tạo lại để so sánh kết quả.", foreground="#555555").grid(row=2, column=1, columnspan=5, sticky="w", pady=(8, 0))

        image_area = ttk.Frame(container)
        image_area.pack(fill="both", expand=True)
        image_area.columnconfigure(0, weight=1)
        image_area.columnconfigure(1, weight=1)
        image_area.rowconfigure(1, weight=1)

        ttk.Label(image_area, text="Ảnh gốc", style="ImageTitle.TLabel").grid(row=0, column=0, pady=(0, 8))
        ttk.Label(image_area, text="Tranh chì", style="ImageTitle.TLabel").grid(row=0, column=1, pady=(0, 8))

        original_frame = ttk.Frame(image_area, relief="solid", borderwidth=1)
        original_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        result_frame = ttk.Frame(image_area, relief="solid", borderwidth=1)
        result_frame.grid(row=1, column=1, sticky="nsew", padx=(7, 0))

        self.original_label = ttk.Label(original_frame, text="Chưa chọn ảnh", anchor="center")
        self.original_label.pack(fill="both", expand=True, padx=8, pady=8)

        self.result_label = ttk.Label(result_frame, text="Chưa có kết quả", anchor="center")
        self.result_label.pack(fill="both", expand=True, padx=8, pady=8)

        status_frame = ttk.Frame(container)
        status_frame.pack(fill="x", pady=(12, 0))

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 12))
        ttk.Label(status_frame, textvariable=self.status).pack(side="right")

    def select_image(self):
        input_path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Ảnh", "*.jpg *.jpeg *.png"), ("Tất cả tệp", "*.*")])

        if not input_path:
            return

        try:
            self.img_rgb = load_image(input_path)
        except Exception as error:
            messagebox.showerror("Không thể mở ảnh", str(error))
            return

        self.sketch = None
        self.show_image(self.img_rgb, self.original_label, "original")
        self.result_label.configure(image="", text="Chưa có kết quả")
        self.process_button.configure(text="Tạo tranh chì", state="normal")
        self.save_button.configure(state="disabled")

        height, width = self.img_rgb.shape[:2]
        filename = Path(input_path).name
        self.status.set(f"Đã chọn {filename} ({width} x {height})")

    def get_parameters(self):
        smoothing_kernel_size, smoothing_sigma = DETAIL_PRESETS[self.detail_level.get()]
        sketch_kernel_size, sketch_sigma = PENCIL_PRESETS[self.pencil_style.get()]
        return smoothing_kernel_size, smoothing_sigma, sketch_kernel_size, sketch_sigma

    def start_processing(self):
        if self.img_rgb is None:
            messagebox.showwarning("Chưa có ảnh", "Hãy chọn ảnh trước khi xử lý")
            return

        try:
            parameters = self.get_parameters()
        except (tk.TclError, ValueError) as error:
            messagebox.showerror("Tham số không hợp lệ", str(error))
            return

        self.open_button.configure(state="disabled")
        self.process_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.progress.start(10)
        self.status.set("Đang tạo tranh chì...")

        worker = threading.Thread(target=self.process_image, args=parameters, daemon=True)
        worker.start()

    def process_image(self, smoothing_kernel_size, smoothing_sigma, sketch_kernel_size, sketch_sigma):
        start_time = time.perf_counter()

        try:
            sketch = pencil_sketch_pipeline(self.img_rgb, smoothing_kernel_size, smoothing_sigma, sketch_kernel_size, sketch_sigma)
        except Exception as error:
            self.root.after(0, lambda message=str(error): self.processing_failed(message))
            return

        elapsed_time = time.perf_counter() - start_time
        self.root.after(0, lambda: self.processing_finished(sketch, elapsed_time))

    def processing_finished(self, sketch, elapsed_time):
        self.sketch = sketch
        self.show_image(self.sketch, self.result_label, "sketch")
        self.progress.stop()
        self.open_button.configure(state="normal")
        self.process_button.configure(text="Tạo lại", state="normal")
        self.save_button.configure(state="normal")
        self.status.set(f"Hoàn thành trong {elapsed_time:.2f} giây")

    def processing_failed(self, message):
        self.progress.stop()
        self.open_button.configure(state="normal")
        self.process_button.configure(state="normal")
        self.save_button.configure(state="disabled")
        self.status.set("Xử lý thất bại")
        messagebox.showerror("Không thể xử lý ảnh", message)

    def show_image(self, image_array, label, image_type):
        image = Image.fromarray(image_array)
        image.thumbnail((520, 520), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        label.configure(image=photo, text="")

        if image_type == "original":
            self.original_photo = photo
        else:
            self.sketch_photo = photo

    def save_result(self):
        if self.sketch is None:
            messagebox.showwarning("Chưa có kết quả", "Hãy tạo tranh chì trước khi lưu")
            return

        output_path = filedialog.asksaveasfilename(title="Lưu tranh chì", defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg")])

        if not output_path:
            return

        try:
            save_image(self.sketch, output_path)
        except Exception as error:
            messagebox.showerror("Không thể lưu ảnh", str(error))
            return

        self.status.set(f"Đã lưu kết quả: {Path(output_path).name}")
        messagebox.showinfo("Lưu thành công", "Tranh chì đã được lưu")

def main():
    root = tk.Tk()
    PencilSketchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
