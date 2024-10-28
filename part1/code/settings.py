from pathlib import Path
import torch
from torchvision import transforms

DATA_PATH = Path("data/content/sorted_movie_posters_paligema")
PCT_IN_TRAIN = 0.7
PCT_IN_VAL = 0.15
BATCH_SIZE = 32
EPOCHS = 1
LR = 1e-3
NUM_WORKERS = 10
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,], std=[0.5,])
])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
CLASS_TO_IDX = {'action': 0, 'animation': 1, 'comedy': 2, 'documentary': 3, 'drama': 4, 'fantasy': 5, 'horror': 6, 'romance': 7, 'science Fiction': 8, 'thriller': 9}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
WEIGHTS_PATH = Path("code/resnet18_weights.pth")
