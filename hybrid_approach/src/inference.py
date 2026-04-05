import numpy as np
import pandas as pd
import joblib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    FINETUNED_MODEL_PATH,
    CENTROID_PATH,
    BP_LABEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    META_MODEL_PATH,
    TEST_DATA_PATH,
    RESULT_PATH
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


print("Loading model")

model = SentenceTransformer(FINETUNED_MODEL_PATH)
meta_model = joblib.load(META_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

centroid_matrix = np.load(CENTROID_PATH)
bp_list = list(np.load(BP_LABEL_PATH))

df = pd.read_excel(TEST_DATA_PATH)
df["category"] = df["category_name"].astype(str)

texts = ["query: " + x for x in df["category"]]

embeddings = model.encode(texts, normalize_embeddings=True)

# pseudo BP
sim_matrix = cosine_similarity(embeddings, centroid_matrix)
df["bp"] = [bp_list[np.argmax(s)] for s in sim_matrix]

bp_centroids = {
    bp_list[i]: centroid_matrix[i]
    for i in range(len(bp_list))
}

df = compute_centroid_signals(df, embeddings, bp_centroids)
df = compute_knn_signal(df, embeddings)
df = compute_cluster_signal(df, embeddings)
df = compute_bp_outlier(df, embeddings)

df[FEATURES] = scaler.transform(df[FEATURES])

X = pd.concat([df[FEATURES], pd.DataFrame(embeddings)], axis=1)

preds = meta_model.predict(X)
probs = meta_model.predict_proba(X)

df["predicted_bp"] = le.inverse_transform(preds)
df["confidence"] = probs.max(axis=1)

df.to_csv(RESULT_PATH, index=False)

print("Done")