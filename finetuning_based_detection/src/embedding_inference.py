import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from config import MODEL_SAVE_PATH, DATA_PATH,EMBEDDING_PATH,CENTROID_PATH,BP_LABEL_PATH


print("Loading trained model")

model = SentenceTransformer(MODEL_SAVE_PATH)


print("Loading dataset")

df = pd.read_excel(DATA_PATH)

df = df.rename(columns={
    "jdmart_catname": "category",
    "BP": "bp"
})


print("Generating embeddings")

texts = ["query: " + str(x) for x in df["category"]]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


np.save(EMBEDDING_PATH, embeddings)


print("Embeddings saved")


# --------------------------------------------------
# COMPUTE BP CENTROIDS
# --------------------------------------------------

print("Computing BP centroids")

bp_centroids = {}

for bp in df["bp"].unique():

    idx = df[df["bp"] == bp].index

    centroid = embeddings[idx].mean(axis=0)

    centroid = centroid / np.linalg.norm(centroid)

    bp_centroids[bp] = centroid


centroid_matrix = np.vstack(list(bp_centroids.values()))

np.save(
    CENTROID_PATH,
    centroid_matrix
)

np.save(
    BP_LABEL_PATH,
    np.array(list(bp_centroids.keys()))
)

print("BP centroids saved")