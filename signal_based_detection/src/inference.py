import numpy as np
import pandas as pd
import joblib
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    MODEL_NAME,
    CENTROID_PATH,
    BP_LABEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    META_MODEL_PATH,
    TEST_RESULT,
    TEST_DATA_PATH
)

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


# =====================================================
# LOAD ARTIFACTS
# =====================================================
print("Loading model + artifacts")

model = SentenceTransformer(MODEL_NAME)
meta_model = joblib.load(META_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

centroid_matrix = np.load(CENTROID_PATH)
bp_list = list(np.load(BP_LABEL_PATH))


# =====================================================
# LOAD DATA (ONLY CATEGORY)
# =====================================================
print("Loading test data")

df = pd.read_excel(TEST_DATA_PATH)

# df = df.rename(columns={
#     "jdmart_catname": "category"
# })

df["category"] = df["category_name"].astype(str)


# =====================================================
# EMBEDDINGS
# =====================================================
print("Generating embeddings")

texts = ["query: " + x for x in df["category"]]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# =====================================================
# INITIAL CENTROID PREDICTION
# =====================================================
print("Initial centroid prediction")

sim_matrix = cosine_similarity(embeddings, centroid_matrix)

initial_bp = []

for sims in sim_matrix:
    best_idx = np.argmax(sims)
    initial_bp.append(bp_list[best_idx])

df["bp"] = initial_bp   # pseudo BP


# =====================================================
# BUILD CENTROID DICT
# =====================================================
bp_centroids = {
    bp_list[i]: centroid_matrix[i]
    for i in range(len(bp_list))
}


# =====================================================
# SIGNAL GENERATION
# =====================================================
print("Computing signals")

df = compute_centroid_signals(df, embeddings, bp_centroids)
df = compute_knn_signal(df, embeddings)
df = compute_cluster_signal(df, embeddings)
df = compute_bp_outlier(df, embeddings)


# =====================================================
# SCALE FEATURES
# =====================================================
print("Scaling features")

df[FEATURES] = scaler.transform(df[FEATURES])


# =====================================================
# PREPARE MODEL INPUT
# =====================================================
embedding_df = pd.DataFrame(embeddings)

X = pd.concat([df[FEATURES], embedding_df], axis=1)


# =====================================================
# FINAL PREDICTION (META MODEL)
# =====================================================
print("Predicting final BP")

preds = meta_model.predict(X)
probs = meta_model.predict_proba(X)

df["predicted_bp"] = le.inverse_transform(preds)
df["confidence"] = probs.max(axis=1)


# =====================================================
# SAVE RESULTS
# =====================================================
df = df.sort_values("confidence", ascending=False)

df.to_csv(TEST_RESULT, index=False)

print("Final results saved:", TEST_RESULT)