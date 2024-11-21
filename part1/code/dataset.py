

import os
import torch
import argparse
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader,random_split
from torchvision.models import ResNet50_Weights
from PIL import Image 
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import os
from torchvision.transforms import InterpolationMode
from settings import BATCH_SIZE,DATA_DIR,NUM_WORKERS


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Définir le répertoire de vos données
# Remplacez par le chemin réel de votre

from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop

def get_loader(path=DATA_DIR,numworkers=NUM_WORKERS,batchsize=BATCH_SIZE):

    # Transformation pour le train dataset avec des augmentations
    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),                # Retourner horizontalement
    transforms.RandomRotation(degrees=15),                 # Rotation aléatoire de +/- 15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Variations de couleur
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Recadrage et zoom
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),  # Zoom et translation
    transforms.ToTensor(),                                  # Convertir en tenseur
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation ImageNet
    ])

    # Transformation pour val/test datasets (sans augmentation)
    val_test_transform = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR), 
    transforms.CenterCrop(224),                                      
    transforms.ToTensor(),  # includes rescaling to [0.0, 1.0]                                   
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                 
                            std=[0.229, 0.224, 0.225]) ])

    # Charger le dataset en utilisant ImageFolder (avec les sous-dossiers comme classes)
    dataset = datasets.ImageFolder(root=path)



    # Calculer la taille de l'ensemble d'entraînement et de test
    train_size = int(0.6 * len(dataset))  # 60% pour l'entraînement
    val_size = int(0.15 * len(dataset))    # 15% pour la validation
    test_size = len(dataset) - train_size - val_size  # 20% pour le test

    print(len(dataset))

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Appliquer les transformations à chaque sous-dataset
    train_dataset.dataset.transform = train_transform  # Transformations avec augmentation pour le train
    val_dataset.dataset.transform = val_test_transform  # Pas d'augmentation pour validation
    test_dataset.dataset.transform = val_test_transform  # Pas d'augmentation pour test


    train_loader = DataLoader(train_dataset, batch_size=batchsize,num_workers=numworkers, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batchsize,num_workers=numworkers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batchsize,num_workers=numworkers, shuffle=False)


    return train_loader,val_loader,test_loader