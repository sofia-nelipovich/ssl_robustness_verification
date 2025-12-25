import torch
import torch.nn.functional as F
import numpy as np
from configs import Config

def test(model, loader):
    loss_log = []
    acc_log = []
    model.eval()

    for data, target in loader:
        data = data.to(Config.DEVICE)
        target = target.to(Config.DEVICE)
        with torch.no_grad():
            out = model(data)
            loss = F.cross_entropy(out, target)
        loss_log.append(loss.item())

        acc = (out.argmax(dim=1) == target).sum() / len(target)
        acc_log.append(acc.item())

    return np.mean(loss_log), np.mean(acc_log)

def train_epoch(model, optimizer, train_loader, scaler=None):
    loss_log = []
    acc_log = []
    model.train()

    for data, target in train_loader:
        data = data.to(Config.DEVICE)
        target = target.to(Config.DEVICE)
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            logits = model(data)
            loss = F.cross_entropy(logits, target)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_log.append(loss.item())
        acc = (logits.argmax(dim=1) == target).sum() / len(target)
        acc_log.append(acc.item())

    return np.mean(loss_log), np.mean(acc_log)

def train(model, optimizer, n_epochs, train_loader, val_loader, scheduler=None, scaler=None, is_val=False):
    train_loss_log, train_acc_log, val_loss_log, val_acc_log = [], [], [], []

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, optimizer, train_loader, scaler)
        val_loss, val_acc = test(model, val_loader)

        train_loss_log.append(train_loss)
        train_acc_log.append(train_acc)

        val_loss_log.append(val_loss)
        val_acc_log.append(val_acc)

        print(f"Epoch {epoch}")
        print(f" train loss: {np.mean(train_loss)}, train acc: {np.mean(train_acc)}")
        print(f" val loss: {val_loss}, val acc: {val_acc}\n")

        if is_val:
            if scheduler is not None:
                scheduler.step(val_acc)
        else:
            if scheduler is not None:
                scheduler.step()

        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f" Learning Rate: {current_lr}")

    return train_loss_log, train_acc_log, val_loss_log, val_acc_log
