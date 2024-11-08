import pickle
from sklearn.cluster import KMeans
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import streamlit as st
import logging
import numpy as np
import json

logger = logging.getLogger("movie-recommender")
logging.basicConfig(level=logging.INFO)

tokenizer = AutoTokenizer.from_pretrained("Supabase/gte-small")
model = AutoModel.from_pretrained("Supabase/gte-small")


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def embed(text: str):
    input_ids = tokenizer(
        text, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**input_ids)
    averaged = average_pool(outputs.last_hidden_state, input_ids["attention_mask"])
    return F.normalize(averaged, p=2, dim=1).numpy()


logger.info("Loading movie clusterer model ...")
with open("./assets/movies_clusters_model.pkl", "rb") as f:
    kmeans: KMeans = pickle.load(f)
logger.info("Finish loading movie clusterer model")

logger.info("Loading movies metadata ...")
movies_df = pd.read_csv("./assets/movies_metadata.csv")
movies_df = movies_df[movies_df["overview"].notna()]
logger.info("Finish loading movies metadata")

logger.info("Loading precalculated movies embeddings ...")
with open("./assets/movies_embeddings_normalize.pkl", "rb") as f:
    movies_embeddings = pickle.load(f)
logger.info("Finish loading precalculated embeddings")

logger.info("Calculating default clusters ...")
movies_clusters = kmeans.predict(movies_embeddings)
logger.info("Finish clustering")

with open("assets/data.json", "r") as f:
    general_data = json.load(f)

st.title("Movie Recommender")

with st.sidebar:
    st.header("Refine your search")
    st.subheader("Choose your preferred genres")
    genre_options = st.multiselect(
        "What kind of genre are you interested in?",
        general_data["movie_genres"],
    )
    st.subheader("Choose your preferred languages")
    language_options = st.multiselect(
        "What kind of languages do you prefer?",
        list(general_data["languages"].keys()),
    )
    st.subheader("Choose your preferred ratings")
    start_rating, end_rating = st.select_slider(
        "Select the range of your desired ratings",
        options=list(range(1, 11)),
        value=[1, 10],
    )
    st.subheader("Top 5 Ranking")
    top5 = st.checkbox("TOP 5")
    st.subheader("Release Date")
    start_release_date, end_release_date = st.select_slider(
        "Select the range for your movie",
        options=list(range(1874, 2021)),
        value=[1874, 2020],
    )
prompt = st.chat_input("What kind of movie do you want to watch?")
if prompt:
    embeded_prompt = embed(prompt)
    prompt_cluster = kmeans.predict(embeded_prompt)[0]
    tmp_df = movies_df.iloc[np.argwhere(movies_clusters == prompt_cluster).reshape(-1)]
    tmp_df = tmp_df[
        (tmp_df["vote_average"] >= start_rating)
        & (tmp_df["vote_average"] <= end_rating)
    ]
    if len(language_options) > 0:
        tmp_df = tmp_df[
            tmp_df["original_language"].isin(
                [general_data["languages"][k] for k in language_options]
            )
        ]
    tmp_df = tmp_df.sort_values(by="vote_average", ascending=False)
    tmp_df = tmp_df[
        (tmp_df["release_date"] >= str(start_release_date))
        & (tmp_df["release_date"] <= str(end_release_date))
    ]
    if len(genre_options) > 0:
        tmp_df = tmp_df[
            tmp_df["genres"].str.contains("|".join(genre_options), regex=True)
        ]
    if top5:
        tmp_df = tmp_df.head(5)
    st.dataframe(
        tmp_df[
            [
                "original_title",
                "overview",
                "genres",
                "original_language",
                "release_date",
                "title",
            ]
        ]
    )
