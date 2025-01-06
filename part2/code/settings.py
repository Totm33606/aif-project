import torch
import torchvision.transforms as transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean, std)
inv_normalize = transforms.Normalize(
    mean=[-m / s for m, s in zip(mean, std)], std=[1 / s for s in std]
)

TRANSFORM = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB_REC = 5
