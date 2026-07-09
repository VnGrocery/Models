# Nguyên lý thư viện và phương pháp huấn luyện

## 1. PyTorch

PyTorch biểu diễn ảnh, nhãn, trọng số và gradient bằng `Tensor`. Trong mỗi batch:

1. Model thực hiện forward và tạo logits.
2. Hàm loss so sánh logits với nhãn thật.
3. Autograd lan truyền ngược để tính gradient.
4. Optimizer cập nhật các trọng số dựa trên gradient.

Tài liệu: https://docs.pytorch.org/docs/stable/

## 2. torchvision và transfer learning

`torchvision.models` cung cấp các CNN đã pretrained trên ImageNet. Dự án thay
lớp phân loại cuối để tạo đúng số output của router hoặc freshness expert.

Hai phương pháp được hỗ trợ:

- `head_only`: đóng băng backbone, chỉ học classifier mới. Nhanh, ít VRAM và phù
  hợp làm baseline hoặc dataset nhỏ.
- `fine_tune`: cập nhật toàn bộ mạng từ trọng số pretrained. Chậm hơn nhưng thích
  nghi tốt hơn với dấu hiệu hư hỏng thực phẩm.

Tài liệu chính thức:
https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

## 3. Các model có thể chọn

- `mobilenet_v3_small`: nhẹ và nhanh nhất, baseline cho thiết bị yếu.
- `mobilenet_v3_large`: cân bằng tốc độ/chất lượng, lựa chọn mặc định.
- `efficientnet_b0`: scale độ sâu, chiều rộng và độ phân giải có hệ thống.
- `resnet18`: residual connection giúp gradient đi qua mạng ổn định, baseline dễ
  giải thích và so sánh.

Không chọn model chỉ theo accuracy train. So sánh trên cùng test split bằng macro
F1, accuracy, thời gian mỗi epoch, số tham số và dung lượng model.

## 4. Dataset và DataLoader

`ImageFolder` suy ra nhãn từ tên thư mục. `DataLoader` tạo batch, shuffle tập
train, đọc ảnh song song bằng workers, prefetch và pin memory để truyền lên GPU.

Trong dự án:

- Train transform có crop, lật và xoay để giảm overfitting.
- Valid/test chỉ resize, crop giữa và normalize.
- 20% tập train được tách cố định bằng seed để test cuối.
- Tập valid dùng chọn checkpoint và early stopping.
- Test chỉ dùng sau khi quá trình chọn model kết thúc.

Tài liệu: https://docs.pytorch.org/docs/stable/data.html

## 5. CrossEntropyLoss

Model trả về logits, không cần softmax trước loss. Cross entropy kết hợp
`LogSoftmax` và negative log-likelihood. Label smoothing giảm việc model quá tự
tin vào một lớp và có thể tăng khả năng khái quát.

## 6. Optimizer

### AdamW

Điều chỉnh learning rate thích nghi theo từng tham số và tách weight decay khỏi
gradient update. Thường hội tụ nhanh, là lựa chọn mặc định.

### SGD

Dùng momentum và Nesterov. Có thể khái quát tốt nhưng nhạy hơn với learning rate
và thường cần nhiều epoch.

## 7. Learning-rate scheduler

- `plateau`: giảm learning rate khi validation loss ngừng cải thiện.
- `cosine`: giảm learning rate theo đường cosine trong tổng số epoch đã định.

ReduceLROnPlateau:
https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

## 8. Automatic Mixed Precision

`torch.autocast` chạy phép toán phù hợp bằng FP16 để tăng tốc và giảm VRAM.
`GradScaler` phóng đại loss/gradient để tránh gradient FP16 quá nhỏ thành zero.
AMP chỉ được bật khi có CUDA.

Tài liệu: https://docs.pytorch.org/docs/stable/notes/amp_examples.html

## 9. Metric đánh giá

- Accuracy: tỷ lệ dự đoán đúng tổng thể.
- Precision: trong các mẫu model dự đoán là một lớp, bao nhiêu mẫu đúng.
- Recall: trong các mẫu thật của một lớp, model tìm được bao nhiêu.
- F1: trung bình điều hòa của precision và recall.
- Macro F1: F1 trung bình của các lớp, mỗi lớp có trọng số bằng nhau.
- Confusion matrix: số lượng nhãn thật–nhãn dự đoán cho từng cặp lớp.

Dataset mất cân bằng không nên chỉ báo cáo accuracy. Macro F1 và confusion matrix
giúp phát hiện model bỏ qua lớp ít mẫu như `HALF-FRESH`.

## 10. Hyperparameter tuning

Tuning phải giữ nguyên train/valid/test split và seed giữa các trial. Mỗi trial
thay một tập cấu hình có kiểm soát: model, phương pháp, optimizer, learning rate,
batch size. Chọn cấu hình theo validation; test chỉ dùng báo cáo cuối để tránh
điều chỉnh theo test.

Lệnh tuning nhanh:

```powershell
python tune.py --task freshness --category cu --epochs 3
```

Kết quả từng trial nằm trong `runs/`; leaderboard nằm ở
`runs/tuning_<timestamp>.csv`.

