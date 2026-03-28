import numpy as np
import pandas as pd
import os

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report

from config import (
    MODEL_SAVE_PATH,
    CENTROID_PATH,
    BP_LABEL_PATH,
    RESULT_PATH,
    DATA_PATH
)

from data_loader import load_dataset


print("Loading model")
model = SentenceTransformer(MODEL_SAVE_PATH)


# _, test_df = load_dataset()
test_df=pd.read_excel(DATA_PATH)

test_df = test_df.rename(columns={
        "jdmart_catname": "category",
        "BP": "bp"
    })

test_df["category"] = test_df["category"].astype(str)

if len(test_df) == 0:
    print("No test data available")
    exit()

# ✅ SAFETY CHECK
if not os.path.exists(CENTROID_PATH):
    raise ValueError("Centroids not found. Run embedding_and_centroid.py first.")


# ===============================
# TEST EMBEDDINGS
# ===============================
print("Generating test embeddings")

test_texts = ["query: " + x for x in test_df["category"]]

test_embeddings = model.encode(
    test_texts,
    normalize_embeddings=True,
    show_progress_bar=True
)


# ===============================
# LOAD CENTROIDS
# ===============================
centroid_matrix = np.load(CENTROID_PATH)
bp_list = list(np.load(BP_LABEL_PATH))


# ===============================
# SIMILARITY
# ===============================
sim_matrix = cosine_similarity(test_embeddings, centroid_matrix)

predicted_bp = []
margin = []

for i, sims in enumerate(sim_matrix):

    current_bp = test_df.iloc[i]["bp"]

    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]
    best_bp = bp_list[best_idx]

    if current_bp in bp_list:
        current_idx = bp_list.index(current_bp)
        current_sim = sims[current_idx]
    else:
        current_sim = 0

    m = best_sim - current_sim

    predicted_bp.append(best_bp)
    margin.append(m)


test_df["predicted_bp"] = predicted_bp
test_df["margin"] = margin


# ===============================
# THRESHOLD DETECTION
# ===============================
THRESHOLD = 0.15

test_df["is_mismatch"] = test_df["margin"] >= THRESHOLD

test_df["suggested_bp"] = test_df["bp"]
test_df.loc[test_df["is_mismatch"], "suggested_bp"] = test_df["predicted_bp"]


print("Total mismatches:", test_df["is_mismatch"].sum())


# ===============================
# CONFUSION MATRIX
# ===============================
print("\nConfusion Matrix:\n")

cm = confusion_matrix(
    test_df["bp"],
    test_df["suggested_bp"],
    labels=bp_list
)

print(cm)


print("\nClassification Report:\n")

print(classification_report(
    test_df["bp"],
    test_df["suggested_bp"]
))


# ===============================
# SAVE RESULTS
# ===============================
test_df = test_df.sort_values("margin", ascending=False)

test_df.to_csv(RESULT_PATH, index=False)

print("Results saved")