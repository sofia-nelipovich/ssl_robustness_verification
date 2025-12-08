import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import DataLoader

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(128, 128)):
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Директория {self.root_dir} не существует")
        
        self.class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        if not self.class_folders:
            raise ValueError(f"Не найдено папок с классами в {self.root_dir}")
        
        self.class_names = [folder.name for folder in self.class_folders]
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.class_to_indices = defaultdict(list)
        
        self._collect_data()
        
        print(f"Датасет загружен: {len(self.image_paths)} изображений, {len(self.class_names)} классов")
    
    def _collect_data(self):
        for class_name, class_idx in self.class_to_idx.items():
            class_folder = self.root_dir / class_name
            
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
            image_files = [f for f in class_folder.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            for img_path in image_files:
                idx = len(self.image_paths)
                self.image_paths.append(img_path)
                self.labels.append(class_idx)
                self.class_to_indices[class_idx].append(idx)
    
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
            
        except Exception as e:
            print(f"Ошибка загрузки изображения {img_path}: {e}")
            dummy_image = torch.zeros(3, *self.target_size)
            return dummy_image, label
    
    def get_class_name(self, label):
        return self.idx_to_class.get(label, "Unknown")
    
    def split_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        indices = list(range(len(self)))
        labels = [self.labels[i] for i in indices]
        
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=labels
        )
        
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        temp_labels = [self.labels[i] for i in temp_idx]
        
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_test_ratio,
            random_state=random_seed,
            stratify=temp_labels
        )
        
        train_dataset = Subset(self, train_idx)
        val_dataset = Subset(self, val_idx)
        test_dataset = Subset(self, test_idx)
        
        return train_dataset, val_dataset, test_dataset


# dataset = ImageDataset(
#     root_dir="/kaggle/input/agrinet-128/Agrinet_128",
#     target_size=(128, 128)
# )

# image, label = dataset[0]
# print(f"Image shape: {image.shape}")
# print(f"Label: {label} - {dataset.get_class_name(label)}")

# train_dataset, val_dataset, test_dataset = dataset.split_dataset(
#     train_ratio=0.7,
#     val_ratio=0.15,
#     test_ratio=0.15
# )

# print(f"\nDataset sizes:")
# print(f"Train: {len(train_dataset)}")
# print(f"Val: {len(val_dataset)}")
# print(f"Test: {len(test_dataset)}")

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=32,
#     shuffle=True
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=32,
#     shuffle=False
# )

# test_loader = DataLoader(
#     test_dataset,
#     batch_size=32,
#     shuffle=False
# )