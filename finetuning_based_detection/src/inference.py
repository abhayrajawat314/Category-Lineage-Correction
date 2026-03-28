import numpy as np
import pandas as pd
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    MODEL_SAVE_PATH,
    CENTROID_PATH,
    BP_LABEL_PATH,
    TEST_RESULT,
    TEST_DATA_PATH
)


# =====================================================
# LOAD MODEL
# =====================================================
print("Loading finetuned model")

model = SentenceTransformer(MODEL_SAVE_PATH)


# =====================================================
# LOAD DATA (ONLY CATEGORY)
# =====================================================
print("Loading dataset")

df = pd.read_excel(TEST_DATA_PATH)

# df = df.rename(columns={
#     "jdmart_catname": "category"
# })

df["category"] = df["category_name"].astype(str)

if len(df) == 0:
    print("Empty dataset")
    exit()


# =====================================================
# SAFETY CHECK
# =====================================================
if not os.path.exists(CENTROID_PATH):
    raise ValueError("Centroids not found. Run embedding_and_centroid.py first.")


# =====================================================
# GENERATE EMBEDDINGS
# =====================================================
print("Generating embeddings")

texts = ["query: " + x for x in df["category"]]

embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# =====================================================
# LOAD CENTROIDS
# =====================================================
centroid_matrix = np.load(CENTROID_PATH)
bp_list = list(np.load(BP_LABEL_PATH))


# =====================================================
# CENTROID PREDICTION
# =====================================================
print("Predicting using centroid similarity")

sim_matrix = cosine_similarity(embeddings, centroid_matrix)

predicted_bp = []
margin = []

for sims in sim_matrix:

    sorted_idx = np.argsort(sims)[::-1]

    best_idx = sorted_idx[0]
    second_idx = sorted_idx[1]

    best_bp = bp_list[best_idx]

    predicted_bp.append(best_bp)

    # confidence = difference between best and second best
    m = sims[best_idx] - sims[second_idx]
    margin.append(m)


df["predicted_bp"] = predicted_bp
df["confidence"] = margin


# =====================================================
# OPTIONAL: FLAG LOW CONFIDENCE
# =====================================================
THRESHOLD = 0.105

df["low_confidence"] = df["confidence"] < THRESHOLD


# =====================================================
# SAVE RESULTS
# =====================================================
df = df.sort_values("confidence", ascending=False)

df.to_csv(TEST_RESULT, index=False)

print("Predictions saved at:", TEST_RESULT)