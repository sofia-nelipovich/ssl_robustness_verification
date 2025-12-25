import torch

class Config:
    DATA_ROOT = "./data/AwA2"
    SAVE_DIR = "./checkpoints"

    SEED = 42
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    METHOD = "SimSiam"
    BACKBONE = "resnet50"
    PROJECTION_DIM = 2048

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    WANDB_PROJECT = "SSL_Backbone_Training"
    WANDB_ENTITY = "your_team_name"
