import torch

class Config:
    DATA_ROOT = "/kaggle/input/agrinet-128/Agrinet_128"
    PRETRAINED_PATH = "/kaggle/input/models/moco_v2_800ep_pretrain.pth.tar"

    SEED = 42
    BATCH_SIZE = 256
    NUM_WORKERS = 4
    NUM_CLASSES = 65
    EPOCHS = 30
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.0001

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
