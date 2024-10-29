import torch
from torchvision import models
from flask import Flask, request
from PIL import Image
import io

from settings import DEVICE, WEIGHTS_PATH, TRANSFORM, IDX_TO_CLASS, NUM_CLASSES

app = Flask(__name__)

# Load the model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)  
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was sent
    if not request.data:
        return "No image provided", 400
    else:
        img_binary = request.data

    # Open the image
    img = Image.open(io.BytesIO(img_binary))
    img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # Apply transformations and add batch dimension

    # Make prediction with the model
    with torch.no_grad():
        outputs = model(img_tensor) # Perform forward pass
        _, predicted_idx = torch.max(outputs, 1)
        predicted_idx = predicted_idx.item()

    predicted_genre = IDX_TO_CLASS.get(predicted_idx, "This genre does not exist!")
    return predicted_genre

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
