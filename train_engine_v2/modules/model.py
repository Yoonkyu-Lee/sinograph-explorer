import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18(num_classes):
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def build_model(name, num_classes):
    if name == "resnet18":
        return build_resnet18(num_classes)
    raise ValueError(f"unknown model: {name}")
