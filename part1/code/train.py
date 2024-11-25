import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from settings import EPOCHS, DEVICE, NUM_CLASSES, FC_LR, BASE_LR, WEIGHTS_PATH, DROPOUT_RATE
from data_utils import get_loaders

model = models.resnet18(weights='IMAGENET1K_V1')
model.fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE),                  
    nn.Linear(model.fc.in_features, NUM_CLASSES)
)
model = model.to(DEVICE)

loaders = get_loaders()
criterion = nn.CrossEntropyLoss(reduction='sum')
base_params = [p for name, p in model.named_parameters() if "fc" not in name]
optimizer = optim.Adam([
    {'params': model.fc.parameters(), 'lr': FC_LR},       # LR for fully connected layer
    {'params': base_params, 'lr': BASE_LR}  # LR for others (pretrained) layers
])

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epochs: int = EPOCHS):
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        total_train_samples = 0
        correct_train = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
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
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
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

if __name__ == "__main__":
    train(model, loaders["train"], loaders["val"], criterion, optimizer)
    torch.save(model.state_dict(), WEIGHTS_PATH)