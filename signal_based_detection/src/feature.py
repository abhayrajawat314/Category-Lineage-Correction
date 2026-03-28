import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from config import LABEL_ENCODER_PATH, EMBEDDING_PATH

from config import DATA_PATH, FEATURE_PATH, SCALER_PATH

from preprocess import normalize_text, linguistic_normalize
from embeddings import generate_embeddings
from centroid_utils import compute_bp_centroids
from signals import (
    compute_knn_signal,
    compute_cluster_signal,
    compute_centroid_signals,
    compute_bp_outlier
)

FEATURES = [
    "sim_best_bp",
    "sim_second_bp",
    "similarity_margin",
    "knn_mismatch_ratio",
    "knn_entropy",
    "cluster_consistency",
    "bp_outlier",
    "current_bp_rank"
]


print("Loading dataset")

df = pd.read_excel(DATA_PATH)

df = df.rename(columns={
    "jdmart_catname":"category",
    "BP":"bp"
})


print("Preprocessing text")

df["clean_category"] = df["category"].apply(normalize_text)

df["normalized_text"] = df["clean_category"].apply(linguistic_normalize)


print("Generating embeddings")

embeddings = generate_embeddings(df["normalized_text"].tolist())

print("Encoding BP labels")

le = LabelEncoder()
df["bp_encoded"] = le.fit_transform(df["bp"])

joblib.dump(le, LABEL_ENCODER_PATH)

print("Computing BP centroids")

bp_centroids = compute_bp_centroids(df, embeddings)


df = compute_centroid_signals(df, embeddings, bp_centroids)

df = compute_knn_signal(df, embeddings)

df = compute_cluster_signal(df, embeddings)

df = compute_bp_outlier(df, embeddings)


print("Normalizing signals")

scaler = StandardScaler()

df[FEATURES] = scaler.fit_transform(df[FEATURES])

joblib.dump(scaler, SCALER_PATH)


df.to_csv(FEATURE_PATH, index=False)

print("Features saved:", FEATURE_PATH)