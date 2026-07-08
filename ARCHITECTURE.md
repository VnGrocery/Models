# Kiến trúc AI phân tầng

## Luồng suy luận

1. **Router** nhận diện một trong năm nhóm: `gia_suc`, `gia_cam`, `rau`, `cu`, `qua`.
2. **Freshness expert** tương ứng phân loại: `FRESH`, `HALF-FRESH`, `SPOILED`.
3. Xác suất của expert được nội suy thành **chỉ số độ tươi ước tính 1–10**.

Mỗi nhóm có expert riêng vì dấu hiệu hư hỏng của thịt, rau, củ và quả khác nhau.

## Cấu trúc dữ liệu

```text
data/
  train/
    gia_suc/
      FRESH/ HALF-FRESH/ SPOILED/
    gia_cam/
    rau/
    cu/
    qua/
  valid/
    ... cùng cấu trúc với train ...
  unclassified/
    train/ valid/   # dữ liệu cũ, chờ gán nhóm
```

Một ảnh chỉ đặt ở một nhóm và một mức độ tươi. Không dùng các biến thể augmentation
của cùng ảnh gốc ở cả train và valid vì sẽ gây rò rỉ dữ liệu.

## Huấn luyện

Sau khi phân loại dữ liệu cũ vào đúng nhóm:

```powershell
python train.py router
python train.py freshness --category gia_suc
python train.py freshness --category gia_cam
python train.py freshness --category rau
python train.py freshness --category cu
python train.py freshness --category qua
```

## Dự đoán

```powershell
python predict.py path\to\image.jpg
```

Không thể chạy pipeline đầy đủ cho đến khi `models/router.pth` và năm expert trong
`models/freshness/` được huấn luyện.
