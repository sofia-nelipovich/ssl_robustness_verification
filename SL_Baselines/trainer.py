import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SupervisedTrainer:
    def __init__(self, model, train_loader, val_loader, device, lr, weight_decay):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.1, patience=5
        )

        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }

    def train(self, epochs, experiment_name):
        best_acc = 0.0
        print(f"\nStart Training: {experiment_name}")

        for epoch in range(epochs):
            self.model.train()
            train_loss, correct, total = 0, 0, 0
            pbar = tqdm(self.train_loader, desc=f"Ep {epoch+1}/{epochs}", leave=False)

            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({'acc': f"{100.*correct/total:.1f}%"})

            epoch_train_loss = train_loss / len(self.train_loader)
            epoch_train_acc = 100. * correct / total

            val_loss, val_acc = self.validate()
            self.scheduler.step(val_loss)

            self.history['train_loss'].append(epoch_train_loss)
            self.history['train_acc'].append(epoch_train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), f"temp_best_{experiment_name}.pth")

        print(f"Finished {experiment_name}. Best Val Acc: {best_acc:.2f}%")
        return self.history, best_acc

    def validate(self):
        self.model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return val_loss / len(self.val_loader), 100. * correct / total
