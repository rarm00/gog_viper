from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import json
import numpy as np
from typing import Tuple, Dict, List

class GameUIDataset(Dataset):
    """Dataset class for game UI elements"""
    
    def __init__(self, 
                 dataset_path: str,
                 transform=None,
                 target_size: Tuple[int, int] = (800, 600)):
        self.dataset_path = Path(dataset_path)
        self.transform = transform or T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Load categories
        with open(self.dataset_path / "categories.json") as f:
            self.categories = json.load(f)
            
        # Load all image paths and annotations
        self.samples = []
        for annotation_file in (self.dataset_path / "annotations").glob("*.json"):
            image_file = self.dataset_path / "images" / f"{annotation_file.stem}.png"
            if image_file.exists():
                with open(annotation_file) as f:
                    annotations = json.load(f)
                self.samples.append((str(image_file), annotations))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        image_path, annotations = self.samples[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        
        # Convert annotations to relative coordinates
        for ann in annotations:
            x, y, w, h = ann['bbox']
            ann['bbox'] = [x/width, y/height, w/width, h/height]
        
        # Transform image
        if self.transform:
            image = self.transform(image)
            
        return image, annotations

def create_dataloaders(dataset_path: str,
                      batch_size: int = 8,
                      num_workers: int = 4) -> Tuple[torch.utils.data.DataLoader, 
                                                   torch.utils.data.DataLoader]:
    """Create training and validation dataloaders"""
    
    # Split dataset into train/val
    dataset = GameUIDataset(dataset_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader