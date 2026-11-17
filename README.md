# Chuyển ảnh thành tranh chì — phần Nguyễn Công Chiến

Dự án nhận ảnh RGB, chuyển xám, làm mịn Gaussian, phát hiện biên Sobel,
tạo lớp tô chì bằng Color Dodge rồi kết hợp để hiển thị/lưu ảnh.

## Chạy nhanh

Mở terminal tại thư mục có `run.py`. Python 3.12 hoặc 3.13 trên Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

Chọn ảnh → giữ cấu hình **Cân bằng / Tự nhiên** → **Tạo tranh chì** → **Lưu kết quả**.
Không cần activate venv. Trên máy đang làm việc, `.venv` đã được chuẩn bị;
bản ZIP không kèm môi trường Python, cần chạy hai lệnh cài đặt ở trên khi giải nén sang máy khác.

## Phần việc đã bổ sung

| Yêu cầu của Nguyễn Công Chiến | Cài đặt |
|---|---|
| Sobel Gx, Gy, magnitude, direction | `src/edge_detection.py` |
| Prewitt Gx, Gy, magnitude, direction | `src/edge_detection.py` |
| Laplacian có dấu và bản đồ cường độ biên | `src/edge_detection.py` |
| Phân ngưỡng cố định | `src/thresholding.py` |
| TP, FP, FN, TN, Precision, Recall, F1, IoU | `src/edge_metrics.py` |
| Kiểm thử số học, ảnh tổng hợp, đối chứng, hồi quy pipeline | `tests/test_edges.py` |
| Thí nghiệm Gaussian/ngưỡng và lưu ảnh trung gian | `comparison/compare_edges.py` |

Đã sửa thêm `src/image_io.py` để mở/lưu được đường dẫn tiếng Việt trên Windows:
NumPy đọc/ghi byte, OpenCV giải mã/mã hóa ảnh. Các thuật toán xử lý vẫn tự code.

Giữ ứng dụng dùng Sobel mặc định. Prewitt/Laplacian phục vụ đối chứng và dùng
được qua hàm Python; không thêm lựa chọn giao diện. Canny, Otsu, morphology
không thuộc phạm vi bản này vì là phần nâng cao/tùy chọn.

## Kiểm tra và tạo kết quả bàn giao

```powershell
.\.venv\Scripts\python.exe -m pytest tests -v
.\.venv\Scripts\python.exe -m comparison.compare_edges
```

Lệnh thứ hai thử 6 ảnh có sẵn × 3 toán tử × 3 Gaussian × 3 ngưỡng = **162 cấu hình**;
thêm **180 cấu hình trên 5 ảnh tổng hợp**. Có thể truyền một ảnh riêng:

```powershell
.\.venv\Scripts\python.exe -m comparison.compare_edges "data\samples\191223.jpg"
```

Kết quả có sẵn trong `results/edges/`:

- `report.md`: bảng tóm tắt và giải thích giới hạn phép đo.
- `opencv_comparison.csv`: kết quả chi tiết đối chiếu OpenCV, mở được trong Excel.
- `synthetic_quality.csv`: chất lượng với biên hình học chuẩn, được dựng độc lập với toán tử.
- `sample_01` đến `sample_06`: ảnh gốc, xám, làm mịn, magnitude, phân ngưỡng 60,
  Gx/Gy Sobel và file hướng gradient radian `.npy`. Thứ tự ảnh theo tên trong CSV.

Ảnh thí nghiệm được thu nhỏ giữ tỉ lệ, cạnh dài tối đa 512 px để chạy nhanh.
Ứng dụng vẫn dùng ảnh gốc. Một lệnh thí nghiệm mới ghi đè báo cáo/CSV cùng tên.

Kết quả đã chạy ngày 17/11/2026: **61 test qua** trên Python 3.13.7,
NumPy 2.2.6, OpenCV 4.12.0 và Pillow 13.0.0. Cả ba toán tử có MAE = 0,
F1 = IoU = 1 so với OpenCV trên 162 cấu hình đã thử (chỉ phản ánh độ khớp).
Đã chạy kiểm tra tự động khởi tạo Tkinter, đọc ảnh đường dẫn tiếng Việt,
tạo/hiển thị tranh 576 × 1024, lưu PNG và đọc lại khớp từng pixel.
Log pytest nằm trong `results/edges/test_results.txt`; ảnh tranh trong `demo_sketch.png`.

## Hiểu phần thuật toán để trình bày

**Sobel/Prewitt:** tính hai đạo hàm có dấu Gx, Gy; độ lớn
`sqrt(Gx**2 + Gy**2)` cho biết độ mạnh, `atan2(Gy, Gx)` cho biết hướng gradient.
Sobel dùng trọng số 1–2–1 theo chiều làm mịn; Prewitt dùng 1–1–1.
Đạo hàm giữ `float32`; chỉ bản đồ đưa vào tranh được chuẩn hóa thành `uint8`.
Hướng gradient là pháp tuyến của biên, không phải hướng chạy dọc theo biên.
Ở pixel có magnitude bằng 0, hướng quy ước bằng 0.

`convolve2d` lật kernel nên Gx/Gy **ngược dấu với OpenCV** theo kernel hiện có;
đã kiểm tra cả dấu và padding reflect101. Magnitude không đổi. Giữ quy ước cũ
giúp phần pipeline của nhóm tiếp tục cho cùng kết quả.

**Laplacian:** kernel `[[0,1,0],[1,-4,1],[0,1,0]]` là đạo hàm bậc hai.
`laplacian_response` giữ dấu; `laplacian_edge_detection` lấy trị tuyệt đối rồi
chuẩn hóa. Bản này đo cường độ đáp ứng, không dò zero-crossing và không có direction.

**Phân ngưỡng:** pixel **lớn hơn** ngưỡng thành 255, các pixel còn lại thành 0.
Thử 30, 60, 90. Ngưỡng thấp giữ nhiều chi tiết và nhiễu; ngưỡng cao có thể mất
nét yếu. Vì chuẩn hóa riêng từng ảnh, ngưỡng 60 không phải cùng mức đạo hàm vật lý
trên mọi ảnh. Biên liên tục vẫn được dùng khi trộn tranh để giữ nét mềm.

**Đánh giá:** TP là biên dự đoán trùng biên chuẩn, FP là biên thừa, FN là biên bỏ sót.
Precision = TP/(TP+FP); Recall = TP/(TP+FN); F1 = 2TP/(2TP+FP+FN);
IoU = TP/(TP+FP+FN). Hai mask đều rỗng về biên được chấm 1; các mẫu số 0 khác trả 0.
Chỉ nhận mask `bool` hoặc `uint8` 0/255 để tránh tính nhầm ảnh xám thành biên.

Đối chiếu OpenCV đo **độ khớp cài đặt**. Biên hình học chuẩn đo chất lượng theo
quy ước hai pixel tại ranh giới; so khớp từng pixel nên biên dày sau Gaussian
có thể bị tính là FP dù vẫn xác định đúng đường bao. Không kết luận rằng Gaussian
luôn tốt hơn chỉ từ một điểm F1. Ảnh thực chưa có nhãn biên để chấm chất lượng vật thể.

## Phần còn lại của cả nhóm

Đã hoàn thiện phần code và bộ đánh giá của Chiến trong phạm vi bản này.
Bộ ảnh hiện có **6 ảnh thực**, chưa đủ **tối thiểu 30 ảnh** theo tài liệu nhóm.
Nhóm cần bổ sung ảnh thực có nguồn, đưa code lên branch `feat/edge-detection`
và nhờ Nguyễn Hữu Khánh review trước khi merge. Bản gốc là ZIP nên chưa có lịch sử Git;
các thao tác push/merge chưa được thực hiện ở đây.
