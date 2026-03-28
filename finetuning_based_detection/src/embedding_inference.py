import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    MODEL_SAVE_PATH,
    EMBEDDING_PATH,
    CENTROID_PATH,
    BP_LABEL_PATH
)

from data_loader import load_dataset


print("Loading trained model")
model = SentenceTransformer(MODEL_SAVE_PATH)


train_df, _ = load_dataset()


# ===============================
# GENERATE EMBEDDINGS (TRAIN)
# ===============================
print("Generating embeddings using trained model")

train_texts = ["query: " + x for x in train_df["category"]]

train_embeddings = model.encode(
    train_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

# ✅ Save directly as numpy (no id needed)
np.save(EMBEDDING_PATH, train_embeddings)

print("Embeddings saved")


# ===============================
# COMPUTE CENTROIDS
# ===============================
print("Computing BP centroids")

bp_centroids = {}

for bp in train_df["bp"].unique():

    idx = train_df[train_df["bp"] == bp].index

    centroid = train_embeddings[idx].mean(axis=0)

    centroid = centroid / np.linalg.norm(centroid)

    bp_centroids[bp] = centroid


bp_list = list(bp_centroids.keys())

centroid_matrix = np.vstack(list(bp_centroids.values()))

np.save(CENTROID_PATH, centroid_matrix)
np.save(BP_LABEL_PATH, np.array(bp_list))

print("Centroids saved")