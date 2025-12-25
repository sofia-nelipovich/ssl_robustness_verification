import torch
import torch.optim as optim
from configs import Config
from dataset import get_dataloaders
from model import get_model
from trainer import train
from utils import plot_results

def main():
    train_loader, val_loader = get_dataloaders()
    model = get_model(freeze_backbone=True)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS
    )

    scaler = torch.amp.GradScaler('cuda')

    tr_loss, tr_acc, val_loss, val_acc = train(
        model,
        optimizer,
        Config.EPOCHS,
        train_loader,
        val_loader,
        scheduler,
        scaler
    )

    plot_results(tr_loss, val_loss, 'Train vs Val Loss', 'Loss')
    plot_results(tr_acc, val_acc, 'Train vs Val Accuracy', 'Accuracy')

if __name__ == "__main__":
    main()
