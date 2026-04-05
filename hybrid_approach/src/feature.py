import pandas as pd
import joblib
import numpy as np

from config import (
    DATA_PATH,
    FEATURE_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
)

from preprocess import linguistic_normalize
from embeddings import generate_embeddings
from centroid_utils import compute_bp_centroids
from signals import (
    compute_knn_signal,
    compute_cluster_signal,
    compute_centroid_signals,
    compute_bp_outlier
)

from sklearn.preprocessing import StandardScaler, LabelEncoder


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
    "jdmart_catname": "category",
    "BP": "bp"
})

# =====================
# PREPROCESS
# =====================
# df["clean"] = df["category"].apply(normalize_text)
df["norm"] = df["category"].apply(linguistic_normalize)

# =====================
# EMBEDDINGS (FINETUNED)
# =====================
embeddings = generate_embeddings(df["norm"].tolist())

# =====================
# LABEL ENCODING
# =====================
le = LabelEncoder()
df["bp_encoded"] = le.fit_transform(df["bp"])
joblib.dump(le, LABEL_ENCODER_PATH)

# =====================
# CENTROIDS
# =====================
bp_centroids = compute_bp_centroids(df, embeddings)

# =====================
# SIGNALS
# =====================
df = compute_centroid_signals(df, embeddings, bp_centroids)
df = compute_knn_signal(df, embeddings)
df = compute_cluster_signal(df, embeddings)
df = compute_bp_outlier(df, embeddings)

# =====================
# SCALE
# =====================
scaler = StandardScaler()
df[FEATURES] = scaler.fit_transform(df[FEATURES])

joblib.dump(scaler, SCALER_PATH)

df.to_csv(FEATURE_PATH, index=False)

print("Features ready")