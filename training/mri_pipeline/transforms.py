"""Tiền xử lý ảnh: resize 224x224, chuẩn hóa ImageNet, augmentation CHỈ cho tập Train."""
from torchvision import transforms as T

from config import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


def get_eval_transforms(img_size: int = IMG_SIZE):
    """Dùng cho Val, Test và demo. Không có augmentation."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_train_transforms(img_size: int = IMG_SIZE):
    """Augmentation nhẹ, giữ đặc trưng giải phẫu:
    - cắt-phóng nhẹ (80-100% diện tích), xoay ±15°, lật ngang, dịch nhẹ
    - chỉnh độ sáng/tương phản nhẹ
    - KHÔNG lật dọc, không biến dạng mạnh."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.RandomRotation(degrees=15)], p=0.5),
        T.RandomApply([T.RandomAffine(degrees=0, translate=(0.05, 0.05))], p=0.3),
        T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Đưa tensor đã chuẩn hóa về ảnh [0,1] để hiển thị."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
