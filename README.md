# NeuroScan MRI

Ứng dụng đồ án phân loại ảnh MRI não thành Glioma, Meningioma, Pituitary Tumor và No Tumor.
Giao diện được giữ từ bản mẫu NeuroScan: Phân tích ảnh, So sánh mô hình, Đánh giá mô hình và Thông tin dự án.

## Chạy trên Mac với ResNet50

Mở thư mục dự án trong VS Code bằng **File → Open Folder**. Terminal cần ở
thư mục chứa `server.py`, `app.py` và `requirements-web.txt`.

1. Cài Python 3.12 nếu máy chưa có bằng `brew install python@3.12`.
2. Với bản lấy từ GitHub, đặt checkpoint `resnet50_best.pth` của nhóm vào
   `models/resnet50/` và đổi tên thành `model.pth`. Cấu hình đã có sẵn trong repo.
   Gói `NeuroScan_ResNet50_Mac.zip` đã được bàn giao kèm checkpoint.
3. Mở `Mo_NeuroScan.command`. Lần đầu chương trình tạo `.venv312`, cài thư viện
   và kiểm tra mô hình. Nếu macOS chặn mở trực tiếp, chạy trong Terminal

   ```sh
   bash Mo_NeuroScan.command
   ```

   Có thể gõ `bash` và một dấu cách, kéo file `.command` từ Finder vào Terminal,
   rồi nhấn Enter. Script tự chuyển vào thư mục dự án.
4. Khi Terminal hiện `Uvicorn running`, giữ Terminal mở và vào
   [NeuroScan tại cổng 8002](http://127.0.0.1:8002).
5. Chọn **Mô hình thật → ResNet50**, tải ảnh MRI JPG/PNG từ tập test của nhóm
   và bấm **Chạy dự đoán**.

Script dùng cổng `8002`. Nếu trước đó đã chạy bản ở cổng `8001`, mở lại đúng
địa chỉ trên để dùng bản vừa khởi động. Dừng máy chủ bằng **Ctrl+C** trong Terminal.
Script mặc định dùng CPU. Có thể chọn thiết bị bằng biến `NEUROSCAN_DEVICE`.

Nếu mục chọn mô hình báo `Thiếu model.pth`, kiểm tra checkpoint nằm tại
`models/resnet50/model.pth` trong chính thư mục đang chạy. Nếu báo thiếu thư viện,
chạy lại lệnh bên dưới rồi mở lại script

```sh
.venv312/bin/python -m pip install -r requirements-web.txt -r requirements-pytorch.txt
```

ResNet50 của nhóm dùng đầu phân loại `Dropout(0.3) → Linear(4)`. Bộ nạp hỗ trợ
cả đầu này và đầu `Linear(4)` chuẩn. Cấu hình sử dụng RGB, resize bilinear
224 × 224, chuẩn hóa ImageNet và thứ tự đầu ra **Glioma → Meningioma → No Tumor →
Pituitary Tumor**, khớp pipeline huấn luyện.

Để xem mục **Đánh giá**, bổ sung `metrics.json` theo định dạng bên dưới.

## Chạy bản chính: giao diện NeuroScan + FastAPI

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-web.txt
python -m uvicorn server:api --host 127.0.0.1 --port 8000
```

Mở http://127.0.0.1:8000. Chọn **Dùng hình mô phỏng → Chạy minh họa kết quả**.
Ba mô hình minh họa là ResNet50, EfficientNetB0 và Swin Transformer; các điểm dự đoán minh họa là giả lập. Swagger ở http://127.0.0.1:8000/docs.
Các API chính: POST /api/predict/{model_id}, POST /api/compare, GET /api/metrics/{model_id}.
Hai API POST nhận bytes của ảnh JPG/PNG, tối đa 10 MB.

## Bàn giao checkpoint PyTorch (.pth)

1. Cài thêm `python -m pip install -r requirements-pytorch.txt`.
2. Tạo từng thư mục riêng, ví dụ `models/resnet50/`, đặt checkpoint tên `model.pth`.
3. Sao chép `model_templates/model_config.pytorch.example.json` thành `model_config.json`
   cạnh checkpoint. Chọn `architecture`: `resnet50`, `efficientnet_b0`, hoặc đúng biến thể Swin
   (`swin_t`, `swin_s`, `swin_b`, `swin_v2_t`, `swin_v2_s`, `swin_v2_b`).
   Điền đúng kích thước, resize, chuẩn hóa và thứ tự nhãn theo notebook.
4. Chạy `python app.py --verify-models`. Tải lại trang và chọn **Mô hình thật**.
   Dùng cùng ảnh kiểm tra để đối chiếu đầu ra với notebook.
5. Thêm `metrics.json` nếu nhóm đã đánh giá trên tập test. Xem cấu trúc bên dưới.

Backend dựng lại **kiến trúc torchvision chuẩn với đầu phân loại 4 lớp**. Checkpoint có thể
là state_dict trực tiếp hoặc nằm trong trường state_dict/model_state_dict.
Nếu nhóm dùng timm, kiến trúc tùy biến hoặc đầu phân loại khác, hãy xuất ONNX
hoặc điều chỉnh bộ dựng mô hình tương ứng; chỉ đổi đuôi file sẽ không chạy được.
Bản web so sánh tối đa ba gói mô hình trong models/.

## Gắn mô hình ONNX (phương án tương thích)

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn server:api --host 127.0.0.1 --port 8000
```

Với Windows PowerShell, kích hoạt bằng `.venv\Scripts\Activate.ps1`.
Tạo một thư mục cho mỗi mô hình, ví dụ `models/resnet50/`:

| File | Yêu cầu |
| --- | --- |
| `model.onnx` | Mô hình nhận batch 1, tensor float32 và trả bốn điểm; nếu xuất kèm external data, đặt các file đi kèm đúng vị trí |
| `model_config.json` | Mô tả đầu vào, thứ tự lớp và kiểu đầu ra |
| `metrics.json` | Tùy chọn. Ma trận số đếm của tập Test độc lập, đúng thứ tự nhãn |

Có thể sao chép hai mẫu trong `model_templates/` vào thư mục của mô hình, bỏ phần
`.example` trong tên file và điền thông số thực từ notebook. Ma trận mẫu rỗng
không được app chấp nhận như kết quả đánh giá.

Ví dụ `model_config.json` **chỉ minh họa định dạng**, không dùng nguyên giá trị nếu khác notebook:

```json
{
  "name": "ResNet50",
  "labels": ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"],
  "size": [224, 224],
  "resize": "bilinear",
  "channels": 3,
  "layout": "NCHW",
  "scale": 0.00392156862745098,
  "mean": [0.485, 0.456, 0.406],
  "std": [0.229, 0.224, 0.225],
  "output_type": "logits",
  "input_name": null,
  "output_name": null
}
```

- `size` là [rộng, cao]. Resize trực tiếp; hỗ trợ bilinear, bicubic, nearest và lanczos.
- Ảnh được chỉnh hướng EXIF và chuyển RGB hoặc xám. Chuẩn hóa theo `(pixel * scale - mean) / std` rồi chuyển bố cục NCHW hoặc NHWC. Nếu ONNX đã chứa bước chuẩn hóa thì dùng scale=1, mean=0 và std=1 cho từng kênh.
- `labels` phải theo đúng thứ tự vector đầu ra của chính mô hình. `output_type` là `logits` nếu cần softmax hoặc `probabilities` nếu mô hình đã trả xác suất.
- Đặt rõ tên input/output nếu model có nhiều output; app mặc định lấy cổng đầu tiên. Nếu notebook dùng crop, letterbox hoặc phép xử lý ngoài hợp đồng trên, bổ sung vào `predict()` trước khi đối chiếu.

Ví dụ `metrics.json` **chỉ minh họa cấu trúc**:

```json
{
  "test_set_id": "brisc-test-v1",
  "labels": ["Glioma", "Meningioma", "Pituitary Tumor", "No Tumor"],
  "confusion_matrix": [[9,1,0,0],[1,9,0,0],[0,0,9,1],[0,0,1,9]]
}
```

Hàng là nhãn thực, cột là nhãn dự đoán. App tính Accuracy, Precision Macro, Recall Macro và F1 Macro từ ma trận; khi mẫu số bằng 0, chỉ số tương ứng lấy 0. Dùng cùng test_set_id cho các mô hình đã đánh giá trên đúng cùng tập Test.

Sau khi chép file, chạy `python3 app.py --check-models` để kiểm tra file và cấu hình. Lệnh này không nạp ONNX. Chạy thêm `python app.py --verify-models` để thử nạp và suy luận trên ảnh tổng hợp; bước này kiểm tra đầu vào/đầu ra, chưa xác nhận độ chính xác. Tải lại trang, chuyển sang **Mô hình thật** và chọn ảnh kiểm tra. Đối chiếu điểm của cả bốn nhãn với notebook trên cùng ảnh trước khi demo. Gắn từng mô hình độc lập, không cần đợi đủ bốn. Nếu thiếu mô hình, thiếu file đánh giá hoặc suy luận lỗi, ứng dụng báo trạng thái tương ứng và không dùng kết quả giả lập để thay thế.

Ứng dụng chỉ chạy localhost. Ảnh không ghi xuống đĩa; ảnh đưa vào chế độ thật được gửi từ trình duyệt đến tiến trình Python trên máy đang chạy app. Đây là ứng dụng nghiên cứu, không dùng để chẩn đoán.

## Mã nguồn trên GitHub

Chạy ứng dụng cục bộ theo hướng dẫn trên. Giữ checkpoint, ảnh MRI và dữ liệu huấn luyện ngoài Git.

## Chạy thử trước buổi demo

- Mở bản chính, kiểm tra ba màn hình Phân tích ảnh, So sánh mô hình và Đánh giá.
- Dùng hình mô phỏng để thử Top-3 và bảng so sánh, sau đó chuyển sang Mô hình thật.
- Chuẩn bị ảnh JPG/PNG được phép chia sẻ từ tập test, ghi lại nhãn thật và kết quả notebook.
- Chạy cùng ảnh qua ba checkpoint, đối chiếu kết quả và mở Confusion Matrix trên tập test.
- Quay một lượt thao tác 60–90 giây sau khi checkpoint thật chạy ổn để dùng dự phòng.

## Kiểm thử phần demo

```sh
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Các test sử dụng một mô hình ONNX bốn đầu ra được tạo tạm để kiểm tra kết nối API,
kiểm tra ảnh PNG/JPEG, ảnh lỗi và quá dung lượng, ảnh sau resize, thứ tự nhãn,
tổng xác suất, so sánh khi thiếu hai checkpoint và trạng thái có/thiếu metrics.
Test này không xác nhận độ chính xác MRI hoặc khả năng nạp checkpoint của nhóm.

Dùng ID thư mục `resnet50`, `efficientnet_b0` và `swin` để bảng so sánh hiển thị
đủ ba vị trí kể cả khi còn thiếu checkpoint. Trình xem ảnh sau tiền xử lý hiển thị
bản đã resize trước khi chuẩn hóa tensor. Chế độ minh họa không hiển thị số liệu
đánh giá nghiên cứu. Mặc định PyTorch ưu tiên CUDA, sau đó MPS rồi CPU.
Nếu gặp phép toán chưa được hỗ trợ trên MPS, khởi động với `NEUROSCAN_DEVICE=cpu`.
