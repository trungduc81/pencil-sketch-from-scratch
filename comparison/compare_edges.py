"""Một lệnh tạo bảng đối chứng và ảnh biên, không thêm giao diện.

Chạy từ thư mục dự án: python -m comparison.compare_edges [duong_dan_anh]
Không truyền ảnh: thử toàn bộ data/samples, không tải thêm dataset.
"""

import argparse
import csv
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image

from src.edge_detection import (
    gradient_magnitude_direction, laplacian_edge_detection, normalize_edges,
    prewitt_edge_detection, sobel_edge_detection, sobel_gradients,
)
from src.edge_metrics import evaluate_edges
from src.grayscale import rgb_to_grayscale
from src.smoothing import gaussian_filter
from src.thresholding import threshold_edges

OPERATORS = {"sobel": sobel_edge_detection, "prewitt": prewitt_edge_detection,
             "laplacian": laplacian_edge_detection}
PRESETS = [(3, 0.8), (5, 1.0), (7, 1.5)]
THRESHOLDS = (30, 60, 90)


def reference_edges(image, operator):
    """Chỉ phần đối chứng dùng OpenCV; cùng padding và kernel tự code."""
    border = cv2.BORDER_REFLECT_101
    if operator == "laplacian":
        return normalize_edges(cv2.Laplacian(image, cv2.CV_32F, ksize=1, borderType=border))
    if operator == "sobel":
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3, borderType=border)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3, borderType=border)
    else:
        kernel = np.array([[-1, 0, 1]] * 3, dtype=np.float32)
        gx = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=border)
        gy = cv2.filter2D(image, cv2.CV_32F, kernel.T, borderType=border)
    # Dùng cùng sqrt và chuẩn hóa để chỉ đo khác biệt của toán tử.
    return normalize_edges(np.sqrt(gx ** 2 + gy ** 2))


def timed(function, *args):
    function(*args)  # Làm nóng một lần, không ghi vào kết quả.
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        result = function(*args)
        samples.append((time.perf_counter() - start) * 1000)
    return result, float(np.median(samples))


def synthetic_cases():
    """Biên chuẩn hai pixel tại bước nhảy, dựng độc lập với toán tử."""
    vertical = np.zeros((96, 96), dtype=np.uint8)
    vertical[:, 48:] = 255
    truth = np.zeros_like(vertical)
    truth[:, 47:49] = 255
    rectangle = np.zeros_like(vertical)
    rectangle[24:72, 24:72] = 255
    outline = np.zeros_like(vertical)
    outline[23:25, 24:72] = outline[71:73, 24:72] = 255
    outline[24:72, 23:25] = outline[24:72, 71:73] = 255
    noisy = np.clip(vertical.astype(np.float32) + np.random.default_rng(42).normal(0, 20, vertical.shape), 0, 255).astype(np.uint8)
    return [("constant", np.full_like(vertical, 128), np.zeros_like(vertical)),
            ("vertical", vertical, truth), ("horizontal", vertical.T.copy(), truth.T.copy()),
            ("rectangle", rectangle, outline), ("noisy_vertical", noisy, truth)]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_signed(response, path):
    maximum = np.max(np.abs(response))
    display = np.full(response.shape, 128, dtype=np.uint8) if maximum == 0 else np.clip(np.rint(128 + response / maximum * 127), 0, 255).astype(np.uint8)
    Image.fromarray(display).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    args = parser.parse_args()
    paths = [args.image] if args.image else sorted(p for p in Path("data/samples").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        parser.error("Không có ảnh trong data/samples; hãy truyền đường dẫn ảnh")

    output = Path("results/edges")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, path in enumerate(paths, 1):
        # Pillow đọc đường dẫn tiếng Việt; giới hạn 512 chỉ trong thực nghiệm.
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            rgb = np.array(image)
        gray = rgb_to_grayscale(rgb)
        folder = output / f"sample_{index:02d}"
        folder.mkdir(exist_ok=True)
        Image.fromarray(rgb).save(folder / "original.png")
        Image.fromarray(gray).save(folder / "gray.png")
        for kernel_size, sigma in PRESETS:
            smoothed = gaussian_filter(gray, kernel_size, sigma)
            for name, detector in OPERATORS.items():
                edges, self_ms = timed(detector, smoothed)
                reference, opencv_ms = timed(reference_edges, smoothed, name)
                mae = float(np.mean(np.abs(edges.astype(np.float32) - reference.astype(np.float32))))
                for threshold in THRESHOLDS:
                    binary = threshold_edges(edges, threshold)
                    scores = evaluate_edges(binary, threshold_edges(reference, threshold))
                    rows.append(dict(image=path.name, height=gray.shape[0], width=gray.shape[1],
                                     operator=name, kernel=kernel_size, sigma=sigma, threshold=threshold,
                                     mae=mae, self_ms=self_ms, opencv_ms=opencv_ms, **scores))
                if kernel_size == 5:
                    Image.fromarray(smoothed).save(folder / "smoothed.png")
                    Image.fromarray(edges).save(folder / f"{name}_magnitude.png")
                    Image.fromarray(threshold_edges(edges)).save(folder / f"{name}_threshold60.png")
            if kernel_size == 5:
                gx, gy = sobel_gradients(smoothed)
                _, direction = gradient_magnitude_direction(gx, gy)
                save_signed(gx, folder / "sobel_gx.png")
                save_signed(gy, folder / "sobel_gy.png")
                np.save(folder / "sobel_direction_radians.npy", direction)

    geometry = []
    for case, image, truth in synthetic_cases():
        for kernel_size, sigma in [(1, 1.0)] + PRESETS:
            smoothed = gaussian_filter(image, kernel_size, sigma)
            for name, detector in OPERATORS.items():
                edges = detector(smoothed)
                for threshold in THRESHOLDS:
                    scores = evaluate_edges(threshold_edges(edges, threshold), truth)
                    geometry.append(dict(case=case, operator=name, kernel=kernel_size,
                                         sigma=sigma, threshold=threshold, **scores))
    write_csv(output / "opencv_comparison.csv", rows)
    write_csv(output / "synthetic_quality.csv", geometry)

    report = ["# Đánh giá phần phát hiện biên — Nguyễn Công Chiến", "",
              f"Môi trường: Python {platform.python_version()}, NumPy {np.__version__}, OpenCV {cv2.__version__}.",
              f"Thực chạy {len(paths)} ảnh có sẵn; mỗi ảnh thử 3 toán tử × 3 Gaussian × 3 ngưỡng.",
              "Ảnh thực nghiệm thu nhỏ giữ tỉ lệ, cạnh dài tối đa 512 px. Ứng dụng chính vẫn xử lý ảnh gốc.",
              "Thời gian: trung vị 3 lần sau 1 lần làm nóng, chỉ tính toán tử và chuẩn hóa, không tính I/O/Gaussian/phân ngưỡng.", "",
              "## Độ khớp với OpenCV", "",
              "Cùng ảnh xám đã làm mịn, padding reflect101, Laplacian ksize=1; cùng công thức chuẩn hóa.",
              "Precision/Recall/F1/IoU so từng pixel với OpenCV chỉ đo độ khớp cài đặt, không chứng minh biên vật thể đúng.", "",
              "| Toán tử | MAE trung bình | F1 thấp nhất | IoU thấp nhất | Tự code (ms) | OpenCV (ms) |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in OPERATORS:
        selected = [r for r in rows if r["operator"] == name]
        report.append(f"| {name} | {np.mean([r['mae'] for r in selected]):.6f} | {min(r['f1'] for r in selected):.6f} | {min(r['iou'] for r in selected):.6f} | {np.mean([r['self_ms'] for r in selected]):.3f} | {np.mean([r['opencv_ms'] for r in selected]):.3f} |")
    report += ["", "## Chất lượng với biên chuẩn hình học", "",
               "5 ảnh tổng hợp: hằng, bước nhảy dọc/ngang, chữ nhật, bước nhảy dọc có nhiễu (seed=42).",
               "Biên chuẩn bước nhảy gồm 2 pixel hai bên ranh giới, tương ứng hỗ trợ của kernel 3×3.",
               "Chữ nhật dùng 2 dải pixel mỗi cạnh; Sobel/Prewitt có thể thêm pixel chéo ở góc, được tính là FP.",
               "Bảng dưới dùng ngưỡng 60, Gaussian 5/sigma 1; CSV chứa cả không làm mịn và các cấu hình khác.", "",
               "| Ảnh tổng hợp | Toán tử | Precision | Recall | F1 | IoU |", "|---|---|---:|---:|---:|---:|"]
    for r in geometry:
        if r["kernel"] == 5 and r["threshold"] == 60:
            report.append(f"| {r['case']} | {r['operator']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | {r['iou']:.4f} |")
    report += ["", "Ở ảnh bước nhảy có nhiễu, ngưỡng 60:", "",
               "| Toán tử | FP không Gaussian | FP Gaussian 5/sigma 1 | F1 không Gaussian | F1 Gaussian 5/sigma 1 |",
               "|---|---:|---:|---:|---:|"]
    for name in OPERATORS:
        raw = next(r for r in geometry if r['case'] == 'noisy_vertical' and r['operator'] == name and r['kernel'] == 1 and r['threshold'] == 60)
        smooth = next(r for r in geometry if r['case'] == 'noisy_vertical' and r['operator'] == name and r['kernel'] == 5 and r['threshold'] == 60)
        report.append(f"| {name} | {raw['fp']} | {smooth['fp']} | {raw['f1']:.4f} | {smooth['f1']:.4f} |")
    report += ["", "## Cách đọc và giới hạn", "",
               "- Ngưỡng tăng loại biên yếu nhưng có thể mất nét mảnh. Gaussian lớn giảm nhiễu và làm rộng biên, nên so pixel chính xác có thể giảm điểm.",
               "- Đáp ứng được chuẩn hóa riêng từng ảnh/toán tử; ngưỡng 60 không biểu thị cùng đạo hàm vật lý giữa các ảnh.",
               "- Laplacian dùng |đạo hàm bậc hai|, không dò zero-crossing hay tính direction.",
               "- Hai mask không có biên: các điểm bằng 1; các mẫu số 0 khác trả 0. Điểm là 0..1.",
               "- Độ khớp OpenCV và chất lượng với biên hình học là hai phép đo riêng. Không có nhãn biên cho ảnh thực.",
               "- Đây là bộ ảnh sẵn có, chưa đáp ứng yêu cầu tối thiểu 30 ảnh thực nghiệm của cả nhóm.",
               "- Ảnh trung gian: sample_XX; Gx/Gy hiển thị xám 128 ứng với 0; direction lưu radian trong .npy.", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Đã ghi {len(rows)} cấu hình đối chứng, {len(geometry)} cấu hình hình học vào {output}")


if __name__ == "__main__":
    main()
