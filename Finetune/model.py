import torch
import torch.nn as nn
from torchvision import models
from configs import Config

class FinetuneModel(nn.Module):
    def __init__(self, backbone, feature_size, num_classes):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def load_pretrained_backbone(backbone, path):
    print(f"Loading weights from: {path}")
    state_dict = torch.load(path, map_location="cpu")

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    backbone_state = {}
    for k, v in state_dict.items():
        if k.startswith("module.encoder_q."):
            new_k = k.replace("module.encoder_q.", "")
        elif k.startswith("encoder_q."):
            new_k = k.replace("encoder_q.", "")
        elif k.startswith("module.backbone_q."):
            new_k = k.replace("module.backbone_q.", "")
        else:
            continue
        backbone_state[new_k] = v

    msg = backbone.load_state_dict(backbone_state, strict=False)
    print("Backbone loaded. Missing keys:", len(msg.missing_keys))
    return backbone

def get_model(freeze_backbone=True):
    backbone = models.resnet50(weights=None)
    feature_size = backbone.fc.in_features
    backbone.fc = nn.Identity()

    if Config.PRETRAINED_PATH:
        backbone = load_pretrained_backbone(backbone, Config.PRETRAINED_PATH)

    model = FinetuneModel(backbone, feature_size, Config.NUM_CLASSES)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model.to(Config.DEVICE)
