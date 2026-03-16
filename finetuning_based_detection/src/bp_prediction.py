import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from config import RESULT_PATH,EMBEDDING_PATH,CENTROID_PATH,DATA_PATH,BP_LABEL_PATH


print("Loading embeddings")

embeddings = np.load(EMBEDDING_PATH)

df = pd.read_csv(
    DATA_PATH
)


print("Computing BP centroids")

centroid_matrix = np.load(
    CENTROID_PATH
)

bp_list = np.load(
    BP_LABEL_PATH
)


print("Computing similarity matrix")

similarity_matrix = cosine_similarity(embeddings, centroid_matrix)


predicted_bp = []
margin = []

for sims in similarity_matrix:

    sorted_idx = np.argsort(sims)[::-1]

    best = sorted_idx[0]
    second = sorted_idx[1]

    predicted_bp.append(bp_list[best])

    margin.append(sims[best] - sims[second])


df["predicted_bp"] = predicted_bp
df["margin"] = margin



print("Computing anomaly scores")

df["confidence"] = df["margin"]

df["anomaly_score"] = 1 - df["confidence"]


threshold = 0.77

df["is_mismatch"] = df["anomaly_score"] >= threshold


df["suggested_bp"] = df["predicted_bp"]

df.loc[~df["is_mismatch"], "suggested_bp"] = df["bp"]


print("Total mismatches detected:", df["is_mismatch"].sum())


df = df.sort_values("anomaly_score", ascending=False)


df.to_csv(RESULT_PATH, index=False)


print("Results saved")