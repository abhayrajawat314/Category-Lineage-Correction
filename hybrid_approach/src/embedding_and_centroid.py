import numpy as np
from sentence_transformers import SentenceTransformer

from config import (
    FINETUNED_MODEL_PATH,
    EMBEDDING_PATH,
    CENTROID_PATH,
    BP_LABEL_PATH
)

from data_loader import load_dataset


print("Loading finetuned model")

model = SentenceTransformer(FINETUNED_MODEL_PATH)


# ===============================
# LOAD DATA
# ===============================
train_df, _ = load_dataset()


# ===============================
# GENERATE EMBEDDINGS
# ===============================
print("Generating embeddings")

texts = ["query: " + str(x) for x in train_df["category"]]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)

np.save(EMBEDDING_PATH, embeddings)

print("Embeddings saved")


# ===============================
# COMPUTE CENTROIDS
# ===============================
print("Computing centroids")

bp_centroids = {}

for bp in train_df["bp"].unique():

    idx = train_df[train_df["bp"] == bp].index

    bp_embeddings = embeddings[idx]

    centroid = bp_embeddings.mean(axis=0)

    # normalize centroid
    centroid = centroid / np.linalg.norm(centroid)

    bp_centroids[bp] = centroid


bp_list = list(bp_centroids.keys())
centroid_matrix = np.vstack(list(bp_centroids.values()))

np.save(CENTROID_PATH, centroid_matrix)
np.save(BP_LABEL_PATH, np.array(bp_list))

print("Centroids saved")