from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
from annoy import AnnoyIndex
from settings import DF_PATH, DF_FILTERED_PATH, ANNOY_PATH, ANNOY_SIZE

device = "cpu"  # Faster on cpu here to get datas...

bert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
bert_matrix_embs = bert_model.embeddings.word_embeddings.weight.to(device)
bert_emb_dim = bert_matrix_embs.shape[1]


def get_embeddings(
    text, tokenizer=bert_tokenizer, embeddings_matrix=bert_matrix_embs, method="mean"
):
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    token_id = inputs["input_ids"].squeeze(0).to(device)
    embs = embeddings_matrix[token_id]
    if method == "mean":
        embs = embs.mean(dim=0).squeeze().detach().cpu().numpy()
    elif method == "sum":
        embs = embs.sum(dim=0).squeeze().detach().cpu().numpy()
    elif method == "cls_token":
        cls_token_idx = 0
        embs = embs[cls_token_idx].squeeze().detach().cpu().numpy()
    elif method == "max":
        embs = embs.max(dim=0).values.squeeze().detach().cpu().numpy()
    return embs


def build_databases(
    df_path=DF_PATH, annoy_path=ANNOY_PATH, df_filtered_path=DF_FILTERED_PATH
):
    # Filter and save df
    movies_dataframe = pd.read_csv(df_path, low_memory=False)
    movies_dataframe = movies_dataframe.filter(
        ["title", "overview", "id", "original_language"]
    )
    movies_dataframe = movies_dataframe[movies_dataframe["original_language"] == "en"]
    movies_dataframe = movies_dataframe.drop(columns=["original_language"])
    movies_dataframe = movies_dataframe.dropna(subset=["title", "overview", "id"])
    movies_dataframe.to_csv(df_filtered_path, index=False)  # Save filtered dataframe

    # Create annoy database
    annoy_index = AnnoyIndex(bert_emb_dim, "angular")

    sampled_movies = movies_dataframe.sample(n=ANNOY_SIZE, random_state=42)

    for i, row in sampled_movies.iterrows():  # Use only some text to build the database
        movie_id = row["id"]
        embeddings = get_embeddings(row["overview"])
        annoy_index.add_item(int(movie_id), embeddings)

    annoy_index.build(10)  # 10 trees

    annoy_index.save(annoy_path)  # Save annoy database


if __name__ == "__main__":
    build_databases()
