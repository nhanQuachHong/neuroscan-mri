# Nguồn huấn luyện ResNet50

`Resnet50_MRI_pipeline_colab.ipynb` giữ nguyên các ô nguồn từ notebook được bàn giao; đầu ra, hình nhúng và số thứ tự thực thi đã được xóa để giảm kích thước trên GitHub.

`resnet50_metrics.original.json` giữ kết quả đánh giá được bàn giao: 985 ảnh test, accuracy 0.9543147208121827, macro F1 0.9465019482424433. File trong `models/resnet50/metrics.json` chuyển ma trận này sang hợp đồng của giao diện.

Thứ tự lớp của lượt đánh giá là `glioma`, `meningioma`, `no_tumor`, `pituitary`; phải giữ đúng thứ tự đầu ra khi dùng checkpoint.

Mã gốc từ `mri_pipeline_phan2_Tri.zip` đã được giải nén nguyên nội dung vào `../training/mri_pipeline/`, gồm train.py, models.py, transforms.py, dataset.py, config.py và các tiện ích. Notebook vẫn cần `split.csv` chính thức và dataset BRISC để tái lập huấn luyện. Khi chạy Colab theo ô nguồn gốc, đặt ZIP mã nguồn trong Drive theo chỉ dẫn Bước 2.

Checkpoint được cài riêng tại `models/resnet50/model.pth`, không đưa vào Git. Chạy `python app.py --verify-models` xác nhận nạp và suy luận trên ảnh tổng hợp. `scripts/verify_resnet50_pipeline.py` so sánh trực tiếp tiền xử lý và xác suất từ checkpoint thật giữa mã huấn luyện và ứng dụng. Báo cáo trong `verification/resnet50_pipeline.json` đã đạt cả bốn đầu vào tổng hợp; chưa đánh giá lại accuracy trên ảnh MRI. Chạy thêm với `--image` trên ảnh test để kiểm tra dữ liệu thật.

