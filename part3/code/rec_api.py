from flask import Flask, request, jsonify
import pandas as pd
from annoy import AnnoyIndex
from data_utils import get_embeddings, bert_emb_dim
from settings import DF_FILTERED_PATH, ANNOY_PATH, NB_REC

app = Flask(__name__)

# Load the dataframe with id - title - overview
df = pd.read_csv(DF_FILTERED_PATH)

# Load the annoy database
annoy_index = AnnoyIndex(bert_emb_dim, "angular")
annoy_index.load(ANNOY_PATH)


@app.route("/predict_close_movies", methods=["POST"])
def predict_close_movies():
    if not request.data:
        return "No text provided", 400
    else:
        text_input = request.data.decode(
            "utf-8"
        )  # Le texte est envoyé dans le corps de la requête

    # Get embs
    text_emb = get_embeddings(text_input)
    ids = annoy_index.get_nns_by_vector(text_emb, NB_REC)
    recommended_titles = df[df["id"].isin(ids)]["title"].tolist()
    return jsonify(recommended_titles)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
