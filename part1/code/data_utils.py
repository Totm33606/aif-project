from pathlib import Path
from typing import Dict
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from settings import DATA_PATH, TRANSFORM, BATCH_SIZE, PCT_IN_TRAIN, PCT_IN_VAL, NUM_WORKERS

def get_loaders(transform: transforms.Compose = TRANSFORM, data_path: Path = DATA_PATH) -> Dict[str, DataLoader]:
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    train_size = int(PCT_IN_TRAIN * len(dataset))
    val_size = int(PCT_IN_VAL * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    
    return {"train": train_loader, "val": val_loader, "test": test_loader}

