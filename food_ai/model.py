import torch.nn as nn
from torchvision import models


def create_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_checkpoint(path, device):
    import torch

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = create_model(len(classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, classes

