"""Khung tạo 3 mô hình với trọng số ImageNet. Người phụ trách từng mô hình có thể
chỉnh phần đầu phân loại (dropout...) nhưng giữ nguyên tên hàm và tên mô hình
để train.py và demo dùng chung."""
import torch.nn as nn

from config import NUM_CLASSES

SUPPORTED = ["resnet50", "efficientnet_b0", "swin_tiny"]


def build_model(name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Trả về (model, tên module lớp phân loại)."""
    if name == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, num_classes))
        return m, "fc"
    if name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m, "classifier"
    if name == "swin_tiny":
        import timm
        m = timm.create_model("swin_tiny_patch4_window7_224", pretrained=pretrained,
                              num_classes=num_classes)
        return m, "head"
    raise ValueError(f"Mô hình '{name}' chưa hỗ trợ. Chọn một trong {SUPPORTED}")


def set_backbone_trainable(model, head_name: str, trainable: bool) -> None:
    """Đóng/mở băng backbone, lớp phân loại luôn được train."""
    for pname, p in model.named_parameters():
        p.requires_grad = trainable or pname.startswith(head_name + ".")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
