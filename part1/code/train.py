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
from settings import EPOCHS,BASE_LR,FC_LR,BATCH_SIZE
from dataset import get_loader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(net, optimizer, train_loader,val_loader ,writer,epochs):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        correct_train = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # FB
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # BW
            loss.backward()
            optimizer.step()
            
            # Do not need to apply softmax manually -> this order will be the same
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()

            train_loss += loss.item()
            total_train_samples += images.size(0)

        
        train_loss = train_loss / total_train_samples
        train_accuracy = correct_train / total_train_samples

        
        # Validation
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        correct_val = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                total_val_samples += images.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
        
        val_loss = val_loss / total_val_samples
        val_accuracy = correct_val / total_val_samples
        
        # Train accuracy could be used to check if the network learns something
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")





epochs = EPOCHS
batch_size = BATCH_SIZE
lr = BASE_LR

# Définir les transformations à appliquer aux images


# Charger le dataset de train
train_loader,val_loader,test_loader=get_loader()

# Récupérer les données de test dans test_loader
test_dataset = test_loader.dataset

# Sauvegarder le dataset complet dans un fichier .pth
test_dataset_path = './test_dataset.pth'  # Chemin pour sauvegarder le dataset
torch.save(test_dataset, test_dataset_path)  # Sauvegarder le dataset complet
print(f"Test dataset sauvegardé ici: {test_dataset_path}")
# Initialiser le writer pour TensorBoard
writer = SummaryWriter(f'runs/movies')



# Nombre de genres dans le dataset
num_genres = len(train_loader.dataset.dataset.classes)
# Charger un modèle pré-entraîné (VGG16 dans cet exemple)
# Charger le modèle VGG16 pré-entraîné
# model = models.vgg16(weights='IMAGENET1K_V1')

# # # Remplacer la couche de classification pour votre nombre de classes

# model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_genres)


# model = models.efficientnet_b0(weights='IMAGENET1K_V1')
# model.classifier = nn.Sequential(
#     nn.Flatten(),
#     nn.Dropout(p=0.5),
#     nn.Linear(model.classifier[1].in_features, len(dataset.classes))
# )

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(
nn.Dropout(p=0.5),                           # Dropout pour éviter le surapprentissage
nn.Linear(model.fc.in_features, 512),         # Couche cachée avec 512 neurones
nn.ReLU(),                                   # Activation ReLU
nn.Dropout(p=0.5),                           # Dropout supplémentaire
nn.Linear(512, num_genres)                   # Couche de sortie avec 'num_genres' neurones
)

model = model.to(device)

# Définir la fonction de perte et l'optimiseur

# # Séparer les paramètres du modèle
# base_params = [p for name, p in model.named_parameters() if "classifier" not in name]

# # Configurer l'optimizer avec des taux d'apprentissage différents
# optimizer = optim.Adam([
#     {'params': model.classifier.parameters(), 'lr': 1e-3},  # LR pour les couches ajoutées
#     {'params': base_params, 'lr': lr},                  # LR pour les couches pré-entraînées
# ])

base_params = [p for name, p in model.named_parameters() if "fc" not in name]
optimizer = optim.Adam([
{'params': model.fc.parameters(), 'lr': FC_LR},       # LR for fully connected layer
{'params': base_params, 'lr': BASE_LR}  # LR for others (pretrained) layers
])


# Entraîner le modèle
train(model, optimizer , train_loader, val_loader,writer=writer, epochs=epochs)



model_save_path = '/content/saved_model.pth'  # Le modèle sera sauvegardé ici
torch.save(model.state_dict(), model_save_path)
print(f"Modèle sauvegardé ici: {model_save_path}")


  