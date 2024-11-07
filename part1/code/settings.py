from pathlib import Path
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DATA_PATH = Path("data/content/sorted_movie_posters_paligema")
PCT_IN_TRAIN = 0.7
PCT_IN_VAL = 0.15
BATCH_SIZE = 32
EPOCHS = 1
BASE_LR = 1e-4
FC_LR = 1e-3
DROPOUT_RATE = 0.5
NUM_WORKERS = 10

# Transform used to train ResNet18 on IMAGENET1K_V1
TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR), 
    transforms.CenterCrop(224),                                      
    transforms.ToTensor(),  # includes rescaling to [0.0, 1.0]                                   
    transforms.Normalize(mean=[0.485, 0.456, 0.406],                 
                         std=[0.229, 0.224, 0.225])
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 10
CLASS_TO_IDX = {'action': 0, 'animation': 1, 'comedy': 2, 'documentary': 3, 'drama': 4, 'fantasy': 5, 'horror': 6, 'romance': 7, 'science Fiction': 8, 'thriller': 9}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
WEIGHTS_PATH = Path("code/resnet18_weights.pth")
