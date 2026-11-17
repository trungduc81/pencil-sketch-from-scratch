# Đánh giá phần phát hiện biên — Nguyễn Công Chiến

Môi trường: Python 3.13.7, NumPy 2.2.6, OpenCV 4.12.0.
Thực chạy 6 ảnh có sẵn; mỗi ảnh thử 3 toán tử × 3 Gaussian × 3 ngưỡng.
Ảnh thực nghiệm thu nhỏ giữ tỉ lệ, cạnh dài tối đa 512 px. Ứng dụng chính vẫn xử lý ảnh gốc.
Thời gian: trung vị 3 lần sau 1 lần làm nóng, chỉ tính toán tử và chuẩn hóa, không tính I/O/Gaussian/phân ngưỡng.

## Độ khớp với OpenCV

Cùng ảnh xám đã làm mịn, padding reflect101, Laplacian ksize=1; cùng công thức chuẩn hóa.
Precision/Recall/F1/IoU so từng pixel với OpenCV chỉ đo độ khớp cài đặt, không chứng minh biên vật thể đúng.

| Toán tử | MAE trung bình | F1 thấp nhất | IoU thấp nhất | Tự code (ms) | OpenCV (ms) |
|---|---:|---:|---:|---:|---:|
| sobel | 0.000000 | 1.000000 | 1.000000 | 10.650 | 4.445 |
| prewitt | 0.000000 | 1.000000 | 1.000000 | 10.755 | 5.827 |
| laplacian | 0.000000 | 1.000000 | 1.000000 | 5.486 | 2.664 |

## Chất lượng với biên chuẩn hình học

5 ảnh tổng hợp: hằng, bước nhảy dọc/ngang, chữ nhật, bước nhảy dọc có nhiễu (seed=42).
Biên chuẩn bước nhảy gồm 2 pixel hai bên ranh giới, tương ứng hỗ trợ của kernel 3×3.
Chữ nhật dùng 2 dải pixel mỗi cạnh; Sobel/Prewitt có thể thêm pixel chéo ở góc, được tính là FP.
Bảng dưới dùng ngưỡng 60, Gaussian 5/sigma 1; CSV chứa cả không làm mịn và các cấu hình khác.

| Ảnh tổng hợp | Toán tử | Precision | Recall | F1 | IoU |
|---|---|---:|---:|---:|---:|
| constant | sobel | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| constant | prewitt | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| constant | laplacian | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| vertical | sobel | 0.5000 | 1.0000 | 0.6667 | 0.5000 |
| vertical | prewitt | 0.5000 | 1.0000 | 0.6667 | 0.5000 |
| vertical | laplacian | 0.3333 | 1.0000 | 0.5000 | 0.3333 |
| horizontal | sobel | 0.5000 | 1.0000 | 0.6667 | 0.5000 |
| horizontal | prewitt | 0.5000 | 1.0000 | 0.6667 | 0.5000 |
| horizontal | laplacian | 0.3333 | 1.0000 | 0.5000 | 0.3333 |
| rectangle | sobel | 0.5026 | 1.0000 | 0.6690 | 0.5026 |
| rectangle | prewitt | 0.5026 | 1.0000 | 0.6690 | 0.5026 |
| rectangle | laplacian | 0.4947 | 0.9789 | 0.6572 | 0.4895 |
| noisy_vertical | sobel | 0.5000 | 1.0000 | 0.6667 | 0.5000 |
| noisy_vertical | prewitt | 0.5000 | 1.0000 | 0.6667 | 0.5000 |
| noisy_vertical | laplacian | 0.3485 | 1.0000 | 0.5168 | 0.3485 |

Ở ảnh bước nhảy có nhiễu, ngưỡng 60:

| Toán tử | FP không Gaussian | FP Gaussian 5/sigma 1 | F1 không Gaussian | F1 Gaussian 5/sigma 1 |
|---|---:|---:|---:|---:|
| sobel | 0 | 192 | 1.0000 | 0.6667 |
| prewitt | 0 | 192 | 1.0000 | 0.6667 |
| laplacian | 949 | 359 | 0.2868 | 0.5168 |

## Cách đọc và giới hạn

- Ngưỡng tăng loại biên yếu nhưng có thể mất nét mảnh. Gaussian lớn giảm nhiễu và làm rộng biên, nên so pixel chính xác có thể giảm điểm.
- Đáp ứng được chuẩn hóa riêng từng ảnh/toán tử; ngưỡng 60 không biểu thị cùng đạo hàm vật lý giữa các ảnh.
- Laplacian dùng |đạo hàm bậc hai|, không dò zero-crossing hay tính direction.
- Hai mask không có biên: các điểm bằng 1; các mẫu số 0 khác trả 0. Điểm là 0..1.
- Độ khớp OpenCV và chất lượng với biên hình học là hai phép đo riêng. Không có nhãn biên cho ảnh thực.
- Đây là bộ ảnh sẵn có, chưa đáp ứng yêu cầu tối thiểu 30 ảnh thực nghiệm của cả nhóm.
- Ảnh trung gian: sample_XX; Gx/Gy hiển thị xám 128 ứng với 0; direction lưu radian trong .npy.
