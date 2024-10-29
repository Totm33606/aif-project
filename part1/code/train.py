import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import EPOCHS, DEVICE, NUM_CLASSES, LR, WEIGHTS_PATH
from data_utils import get_loaders

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

loaders = get_loaders()
criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=LR)

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int = 5):
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        total_train_samples = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # FB
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # BW
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total_train_samples += images.size(0)

        
        train_loss = running_loss / total_train_samples
        
        # Validation
        model.eval()
        val_loss = 0.0
        total_val_samples = 0
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                total_val_samples += images.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
        
        val_loss = val_loss / total_val_samples
        val_accuracy = correct / total_val_samples
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

train(model, loaders["train"], loaders["val"], criterion, optimizer, EPOCHS)
torch.save(model.state_dict(), WEIGHTS_PATH)