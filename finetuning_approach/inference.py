# =========================================================
# INSTALL
# =========================================================
# pip install transformers torch pandas scikit-learn openpyxl


# =========================================================
# IMPORT
# =========================================================
import pandas as pd
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity



print("===== STARTING EMBEDDING PIPELINE =====")


# =========================================================
# LOAD MODEL
# =========================================================
print("Loading fine-tuned model...")

model_path = "bp_embedding_model2"

tokenizer = AutoTokenizer.from_pretrained(model_path)
encoder = AutoModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder.to(device)
encoder.eval()

print("Model loaded.")



# =========================================================
# LOAD DATA
# =========================================================
print("Loading dataset...")

df = pd.read_excel("Sample_data.xlsx")

print("Rows:", len(df))



# =========================================================
# EMBEDDING FUNCTION
# =========================================================
def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=16
    )

    inputs = {k:v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = encoder(**inputs)

    # CLS token embedding
    embedding = outputs.last_hidden_state[:,0,:]

    return embedding.cpu().numpy()[0]



# =========================================================
# CREATE CATEGORY EMBEDDINGS
# =========================================================
print("Generating category embeddings...")

df["embedding"] = df["jdmart_catname"].astype(str).apply(get_embedding)

print("Category embeddings created.")



# =========================================================
# CREATE BASE PARENT EMBEDDINGS
# =========================================================
print("Computing BP centroids...")

bp_embeddings = {}

# =========================================================
# CREATE BASE PARENT EMBEDDINGS (DIRECT)
# =========================================================
print("Generating BP embeddings...")

bp_embeddings = {}

unique_bps = df["BP"].unique()

for bp in unique_bps:

    emb = get_embedding(bp)

    bp_embeddings[bp] = emb

print("BP embeddings ready.")



# =========================================================
# FIND BEST BP USING SIMILARITY
# =========================================================
print("Computing similarities...")

predicted_bp = []
similarity_score = []

for emb in df["embedding"]:

    sims = {}

    for bp, bp_vec in bp_embeddings.items():

        sim = cosine_similarity(
            emb.reshape(1,-1),
            bp_vec.reshape(1,-1)
        )[0][0]

        sims[bp] = sim

    best_bp = max(sims, key=sims.get)

    predicted_bp.append(best_bp)
    similarity_score.append(sims[best_bp])


df["predicted_bp"] = predicted_bp
df["similarity"] = similarity_score



# =========================================================
# DETECT MISMATCHES
# =========================================================
df["is_mismatch"] = df["BP"] != df["predicted_bp"]


print("Total mismatches detected:", df["is_mismatch"].sum())



# =========================================================
# SAVE RESULTS
# =========================================================
df.to_excel("lineage_detection_results3.xlsx", index=False)

print("Results saved.")


print("===== PIPELINE COMPLETE =====")