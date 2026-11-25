from data import build_transform, CUBDataset, PKBatchSampler, make_train_loader

if __name__ == "__main__":
    # Thay bằng đường dẫn thực tế tới thư mục CUB_200_2011 của bạn
    DATA_ROOT = "./CUB_200_2011"

    print("=" * 60)
    print("1. KIỂM TRA CẤU TRÚC CỦA CUBDataset")
    print("=" * 60)
    train_tf = build_transform(train=True, size=224)
    test_tf = build_transform(train=False, size=224)

    train_ds = CUBDataset(root=DATA_ROOT, split="train", transform=train_tf)
    test_ds = CUBDataset(root=DATA_ROOT, split="test", transform=test_tf)

    print(f"Tổng số ảnh Train (lớp 1-100): {len(train_ds)}")
    print(f"Tổng số ảnh Test (lớp 101-200): {len(test_ds)}")
    print(f"Thư mục ảnh gốc (img_dir): {train_ds.img_dir}")

    # In 3 mẫu đầu tiên trong danh sách paths và labels của train_ds
    print("\n--- 3 phần tử đầu tiên trong train_ds.paths và train_ds.labels ---")
    for i in range(3):
        print(f"  Index {i}: label={train_ds.labels[i]} | path={train_ds.paths[i].name}")

    # Gọi __getitem__(0) để xem dữ liệu thực tế đưa vào model
    img_tensor, label = train_ds[0]
    print("\n--- Kết quả khi gọi train_ds[0] ---")
    print(f"  Kiểu dữ liệu ảnh: {type(img_tensor)}, Shape: {img_tensor.shape}, Dtype: {img_tensor.dtype}")
    print(f"  Min/Max pixel (sau Normalize): {img_tensor.min():.2f} / {img_tensor.max():.2f}")
    print(f"  Nhãn (label): {label}")

    print("\n" + "=" * 60)
    print("2. KIỂM TRA CẤU TRÚC CỦA PKBatchSampler")
    print("=" * 60)
    P, K = 4, 3  # Dùng P=4, K=3 cho dễ nhìn (thực tế thường là 16x4)
    sampler = PKBatchSampler(train_ds.labels, p=P, k=K)

    print(f"Số batch mỗi epoch (num_batches): {len(sampler)}")
    print(f"Danh sách các class ID có trong sampler (10 lớp đầu): {sampler.classes[:10]}")

    # In cấu trúc của dictionary self.by_class (chỉ lấy 3 key đầu tiên)
    print("\n--- Cấu trúc từ điển sampler.by_class (3 key đầu) ---")
    for cls_id in list(sampler.by_class.keys())[:3]:
        img_indices = sampler.by_class[cls_id]
        print(f"  Class {cls_id}: có {len(img_indices)} ảnh -> 5 index đầu: {img_indices[:5]}")

    # Lấy thử 1 batch index từ sampler
    first_batch_indices = next(iter(sampler))
    print(f"\n--- 1 Batch index sinh ra từ sampler (P={P}, K={K} -> tổng {len(first_batch_indices)} ảnh) ---")
    print(f"  Indices: {first_batch_indices}")
    # Đối chiếu ngược lại xem các index này thuộc lớp nào
    batch_labels = [train_ds.labels[idx] for idx in first_batch_indices]
    print(f"  Labels tương ứng: {batch_labels}")

    print("\n" + "=" * 60)
    print("3. KIỂM TRA DATA LOADER HOÀN CHỈNH")
    print("=" * 60)
    # Đặt workers=0 khi debug trên Windows/Mac để tránh lỗi multiprocessing
    train_loader = make_train_loader(train_ds, p=P, k=K, workers=0)

    images, labels = next(iter(train_loader))
    print(f"Batch Tensor Images shape: {images.shape}  (Batch_size={P * K}, Channels=3, H=224, W=224)")
    print(f"Batch Tensor Labels:       {labels.tolist()}")