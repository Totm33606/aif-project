import torch
import torch.nn as nn
from torchvision import models
from flask import Flask, request, jsonify
from PIL import Image
import io
import joblib

from settings import (
    DEVICE,
    WEIGHTS_PATH,
    TRANSFORM,
    IDX_TO_CLASS,
    NUM_CLASSES,
    DROPOUT_RATE,
    ANOMALY_WEIGHTS_PATH,
)

app = Flask(__name__)

# Load the classifier
classifier = models.resnet18(weights=None)
classifier.fc = nn.Sequential(
    nn.Dropout(p=DROPOUT_RATE), nn.Linear(classifier.fc.in_features, NUM_CLASSES)
)
classifier.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
classifier.to(DEVICE)
classifier.eval()

# Load the anomaly detector
anomaly_detector = joblib.load(ANOMALY_WEIGHTS_PATH)

# Load the features extractor
features_extractor = models.resnet18(weights="IMAGENET1K_V1")
features_extractor = nn.Sequential(*list(features_extractor.children())[:-1])
features_extractor = features_extractor.to(DEVICE)
features_extractor.eval()


@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image was sent
    if not request.data:
        return "No image provided", 400
    else:
        img_binary = request.data

    # Open the image
    img = Image.open(io.BytesIO(img_binary))
    img_tensor = (
        TRANSFORM(img).unsqueeze(0).to(DEVICE)
    )  # Apply transformations and add batch dimension

    with torch.no_grad():
        # Check if it's an anomaly
        features = features_extractor(img_tensor)
        outputs_anomaly = anomaly_detector.predict(
            features.squeeze(2).squeeze(2).cpu().numpy()
        ).item()
        if outputs_anomaly == 1:
            return jsonify("Error, this image is not a movie poster!")

        # Make prediction with the model
        outputs = classifier(img_tensor)  # Perform forward pass
        _, predicted_idx = torch.max(outputs, 1)
        predicted_idx = predicted_idx.item()

    predicted_genre = IDX_TO_CLASS.get(predicted_idx, "This genre does not exist!")
    return predicted_genre


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
