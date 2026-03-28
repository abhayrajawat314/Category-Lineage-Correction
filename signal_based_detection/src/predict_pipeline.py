import pandas as pd
import joblib
import numpy as np

from config import (
    FEATURE_PATH,
    META_MODEL_PATH,
    FINAL_RESULT,
    EMBEDDING_PATH,
    LABEL_ENCODER_PATH
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
model = joblib.load(META_MODEL_PATH)

print("Loading label encoder")
le = joblib.load(LABEL_ENCODER_PATH)

print("Loading features")
df = pd.read_csv(FEATURE_PATH)

print("Loading embeddings")
embeddings = np.load(EMBEDDING_PATH)

embedding_df = pd.DataFrame(embeddings)

# ---------------------------------------
# COMBINE SIGNALS + EMBEDDINGS
# ---------------------------------------

X = pd.concat([df[FEATURES], embedding_df], axis=1)

print("Predicting BP")

preds = model.predict(X)

# decode labels
df["predicted_bp"] = le.inverse_transform(preds)

probs = model.predict_proba(X)
df["confidence"] = probs.max(axis=1)

# ---------------------------------------
# SAVE RESULTS
# ---------------------------------------

df.to_csv(FINAL_RESULT, index=False)

print("Prediction results saved:", FINAL_RESULT)