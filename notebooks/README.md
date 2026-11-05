# Nguồn huấn luyện ResNet50

`Resnet50_MRI_pipeline_colab.ipynb` giữ nguyên các ô nguồn từ notebook được bàn giao; đầu ra, hình nhúng và số thứ tự thực thi đã được xóa để giảm kích thước trên GitHub.

`resnet50_metrics.original.json` giữ kết quả đánh giá được bàn giao: 985 ảnh test, accuracy 0.9543147208121827, macro F1 0.9465019482424433. File trong `models/resnet50/metrics.json` chuyển ma trận này sang hợp đồng của giao diện.

Thứ tự lớp của lượt đánh giá là `glioma`, `meningioma`, `no_tumor`, `pituitary`; phải giữ đúng thứ tự đầu ra khi dùng checkpoint.

Notebook còn phụ thuộc `mri_pipeline_phan2_Tri.zip` (gồm train.py, models.py, transforms.py, dataset.py và các tiện ích), `split.csv` chính thức và dataset BRISC. Các file mã huấn luyện trong ZIP chưa được bàn giao vào repo này, nên chỉ thêm notebook chưa đủ tái lập huấn luyện hoặc xác nhận toàn bộ tiền xử lý. Không suy ra các phép xử lý ảnh còn thiếu từ tên kiến trúc.

Checkpoint phải được bàn giao riêng, đặt tại `models/resnet50/model.pth`. Chạy `python app.py --verify-models` chỉ xác nhận nạp và suy luận trên ảnh tổng hợp, không xác nhận accuracy hoặc đồng nhất với notebook. Đối chiếu xác suất bốn lớp trên cùng ảnh test bằng chính mã huấn luyện trước khi báo kết quả demo thật.

