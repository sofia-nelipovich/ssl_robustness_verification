import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from configs import Config

class ImageDataset(Dataset):
    def __init__(self, root_dir, target_size=(128, 128)):
        self.root_dir = Path(root_dir)
        self.target_size = target_size

        if not self.root_dir.exists():
            raise FileNotFoundError(f"{self.root_dir} does not exist")

        self.class_folders = sorted(
            [d for d in self.root_dir.iterdir() if d.is_dir()]
        )
        if not self.class_folders:
            raise ValueError("No classes found")

        self.class_names = [d.name for d in self.class_folders]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}

        self.image_paths = []
        self.labels = []

        self._collect_data()
        print(f"Loaded {len(self.image_paths)} images")

    def _collect_data(self):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
        for class_name, class_idx in self.class_to_idx.items():
            folder = self.root_dir / class_name
            for img_path in folder.iterdir():
                if img_path.suffix.lower() in exts:
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return img, label

class TransformDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def get_dataloaders():
    base_dataset = ImageDataset(root_dir=Config.DATA_ROOT)

    indices = np.arange(len(base_dataset))
    labels = np.array(base_dataset.labels)

    train_idx, temp_idx = train_test_split(
        indices, train_size=0.7, stratify=labels, random_state=Config.SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, train_size=0.5, stratify=labels[temp_idx], random_state=Config.SEED
    )

    train_transform, val_transform = get_transforms()

    train_dataset = TransformDataset(Subset(base_dataset, train_idx), train_transform)
    val_dataset = TransformDataset(Subset(base_dataset, val_idx), val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader
