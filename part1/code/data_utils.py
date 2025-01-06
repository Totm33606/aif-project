import os
from pathlib import Path
from typing import Dict, Tuple, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.model_selection import train_test_split
from PIL import Image

from settings import (
    DATA_PATH,
    TRANSFORM,
    BATCH_SIZE,
    PCT_IN_TRAIN,
    PCT_IN_VAL,
    NUM_WORKERS,
    DEVICE,
    ANOMALY_DATA_PATH,
)


def get_classifier_loaders(
    transform: transforms.Compose = TRANSFORM, data_path: Path = DATA_PATH
) -> Dict[str, DataLoader]:
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    train_size = int(PCT_IN_TRAIN * len(dataset))
    val_size = int(PCT_IN_VAL * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def no_anomaly_ft_extraction(
    loader: DataLoader, label: int, features_extractor: nn.Module
) -> Tuple[List[np.array], List[int]]:
    images, labels = [], []
    with torch.no_grad():
        for img, _ in loader:
            img = img.to(DEVICE)
            embs = features_extractor(img)
            images.append(embs.squeeze().cpu().numpy())
            labels.append(label)
    return images, labels


def anomaly_ft_extraction(
    label: int,
    features_extractor: nn.Module,
    folder: Path = ANOMALY_DATA_PATH,
    transform: transforms.Compose = TRANSFORM,
    size: int = 250,
) -> Tuple[List[np.array], List[int]]:
    images, labels = [], []
    count = 0
    with torch.no_grad():
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith((".png", ".jpg", ".jpeg")):
                    filepath = os.path.join(root, file)
                    try:
                        image = Image.open(filepath).convert("RGB")
                        image = transform(image)
                        image = image.unsqueeze(0).to(DEVICE)
                        embs = features_extractor(image)
                        images.append(embs.squeeze().cpu().numpy())
                        labels.append(label)
                        count += 1

                        if count == size:
                            return images, labels
                    except Exception as e:
                        print(f"Error loading {file}: {e}")


def get_anomaly_datas(
    transform: transforms.Compose = TRANSFORM,
    path_no_anomalies: Path = DATA_PATH,
    path_anomalies: Path = ANOMALY_DATA_PATH,
) -> Tuple[Any]:
    features_extractor = models.resnet18(weights="IMAGENET1K_V1")
    features_extractor = nn.Sequential(*list(features_extractor.children())[:-1])
    features_extractor = features_extractor.to(DEVICE)
    features_extractor.eval()

    # The images without anomalies
    dataset_posters = datasets.ImageFolder(root=path_no_anomalies, transform=transform)
    loader_posters = DataLoader(dataset_posters, batch_size=1, shuffle=False)
    images_posters, labels_posters = no_anomaly_ft_extraction(
        loader_posters, 0, features_extractor
    )

    # The images with anomalies (ie random images)
    number_anomalies = int(0.05 * len(images_posters))
    images_no_posters, labels_no_posters = anomaly_ft_extraction(
        1,
        features_extractor,
        folder=path_anomalies,
        transform=transform,
        size=number_anomalies,
    )

    X = np.vstack(images_posters + images_no_posters)
    y = np.array(labels_posters + labels_no_posters)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - PCT_IN_TRAIN - PCT_IN_VAL), random_state=42, stratify=y
    )  # No validation datas needed here

    return X_train, X_test, y_train, y_test
