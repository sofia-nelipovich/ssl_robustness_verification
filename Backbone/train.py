import torch
import torch.optim as optim
import wandb
from tqdm import tqdm
from configs import Config
from dataset import get_dataloader
from model import get_ssl_model
from utils import set_seed, save_checkpoint

def train():
    set_seed(Config.SEED)

    wandb.init(project=Config.WANDB_PROJECT, entity=Config.WANDB_ENTITY)

    dataloader = get_dataloader()
    model = get_ssl_model()

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    model.train()

    for epoch in range(Config.EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")

        for images, _ in progress_bar:
            x1, x2 = images[0].to(Config.DEVICE), images[1].to(Config.DEVICE)

            loss = model(x1, x2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(dataloader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % 10 == 0:
            save_path = f"{Config.SAVE_DIR}/{Config.METHOD}_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, save_path)

    wandb.finish()

if __name__ == "__main__":
    train()
