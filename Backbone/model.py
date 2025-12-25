import torch
import torch.nn as nn
from torchvision import models
from ak_ssl import SimSiam, MoCo
from configs import Config

def get_ssl_model():
    backbone = models.resnet50(weights=None)

    if Config.METHOD == "SimSiam":
        model = SimSiam(
            backbone=backbone,
            image_size=128
        )
    elif Config.METHOD == "MoCo":
        model = MoCo(
            backbone=backbone,
            image_size=128
        )
    else:
        raise ValueError(f"Unknown method: {Config.METHOD}")

    return model.to(Config.DEVICE)
