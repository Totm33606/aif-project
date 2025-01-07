import torch
import torch.nn as nn
from torchvision import models
from flask import Flask, request, send_file
from PIL import Image
import io
import pandas as pd
from annoy import AnnoyIndex
import zipfile

from settings import DEVICE, TRANSFORM, NB_REC, DF_FT_PATH, ANNOY_PATH

app = Flask(__name__)

# Load the dataframe with index - features
df = pd.read_csv(DF_FT_PATH)

# Load the annoy database
dim = 576
annoy_index = AnnoyIndex(dim, "angular")
annoy_index.load(ANNOY_PATH)

# Load the model
mobilenet = models.mobilenet_v3_small(weights="IMAGENET1K_V1")
model = nn.Sequential(mobilenet.features, nn.AvgPool2d((5, 5)), nn.Flatten())
model.to(DEVICE)
model.eval()


@app.route("/predict_close_movies", methods=["POST"])
def predict_close_movies():
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

    # Make prediction with the model
    with torch.no_grad():
        query_vector = model(img_tensor)  # Perform forward pass
        indices = annoy_index.get_nns_by_vector(
            query_vector.squeeze(0).cpu().numpy(), NB_REC
        )
        paths = df.iloc[indices]["path"]
        file_io = io.BytesIO()
        with zipfile.ZipFile(file_io, "w") as zipf:
            for idx in range(NB_REC):
                image_path = paths.iloc[idx]
                img_to_send = Image.open(image_path)
                img_byte_arr = io.BytesIO()
                img_to_send.save(img_byte_arr, format="JPEG")
                zipf.writestr(f"image_{idx + 1}.jpg", img_byte_arr.getvalue())
        file_io.seek(0)
        return send_file(file_io, mimetype="application/zip")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
