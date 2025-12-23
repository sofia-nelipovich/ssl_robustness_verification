import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import shutil
import os


from dataset import ImageDataset
from trainer import SupervisedTrainer
from utils import plot_all_experiments

IMBALANCED_DATA_PATH = "path/to/your/dataset"
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
NUM_CLASSES = 50


HYPERPARAMS_LIST = [
    {"lr": 1e-3, "wd": 1e-4},
    {"lr": 5e-4, "wd": 1e-4},
    {"lr": 1e-4, "wd": 1e-4},
    {"lr": 1e-3, "wd": 1e-2},
    {"lr": 1e-4, "wd": 1e-2},
]

def get_model(num_classes):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, num_classes)
    return model.to(DEVICE)

def main():
    print(f"Using device: {DEVICE}")

    print("Loading Dataset...")
    dataset = ImageDataset(root_dir=IMBALANCED_DATA_PATH)

    train_ds, val_ds, test_ds = dataset.split_dataset(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    all_histories = {}
    global_best_acc = 0.0
    best_params_name = ""

    print(f"Starting Hyperparameter Search with {len(HYPERPARAMS_LIST)} configurations...")

    for params in HYPERPARAMS_LIST:
        lr = params["lr"]
        wd = params["wd"]
        exp_name = f"lr_{lr}_wd_{wd}"

        model = get_model(NUM_CLASSES)

        trainer = SupervisedTrainer(
            model, train_loader, val_loader, DEVICE, lr=lr, weight_decay=wd
        )

        history, run_best_acc = trainer.train(epochs=EPOCHS, experiment_name=exp_name)
        all_histories[exp_name] = history

        if run_best_acc > global_best_acc:
            global_best_acc = run_best_acc
            best_params_name = exp_name
            print(f"New Global Best Found! ({exp_name}: {run_best_acc:.2f}%)")
            shutil.copy(f"temp_best_{exp_name}.pth", "sl_imbalanced_best.pth")

    print("="*30)
    print(f"Search Finished.")
    print(f"Best Configuration: {best_params_name}")
    print(f"Best Validation Accuracy: {global_best_acc:.2f}%")
    print(f"Best model saved to 'sl_imbalanced_best.pth'")
    print("="*30)

    plot_all_experiments(all_histories)


if __name__ == "__main__":
    main()
