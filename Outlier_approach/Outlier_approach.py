# import re
# import numpy as np
# import pandas as pd
# import spacy

# from sentence_transformers import SentenceTransformer
# from sklearn.preprocessing import normalize
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.neighbors import NearestNeighbors
# import hdbscan

# nlp = spacy.load("en_core_web_sm")

# NEUTRAL_WORDS = {
#     "for", "with", "on", "at", "to", "by", "of", "in", "upto", "and"
# }

# GENERIC_TERMS = {
#     "service", "services",
#     "center", "centers", "centre", "centres",
#     "clinic", "clinics",
#     "shop", "shops",
#     "store", "stores",
#     "class","classes",
#     "company","companies",
#     "hub","hubs","station",
#     "cafe","stations","repair"
# }

# df = pd.read_excel("Sample_data.xlsx")

# df = df.rename(columns={
#     "jdmart_catname": "category",
#     "BP": "bp"
# })

# def normalize_text(text: str) -> str:
#     text = str(text).lower()
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# def linguistic_normalize(text: str) -> str:
#     doc = nlp(text)
#     tokens = []

#     for token in doc:
#         lemma = token.lemma_.lower()

#         if lemma in NEUTRAL_WORDS or lemma in GENERIC_TERMS:
#             continue

#         if token.pos_ in {"NOUN", "ADJ", "PROPN"}:
#             tokens.append(lemma)

#     return " ".join(tokens)

# df["clean_category"] = df["category"].apply(normalize_text)
# # df["normalized_text"] = df["clean_category"].apply(linguistic_normalize)



# print("Loading transformer model...")

# model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# print("Generating embeddings...")

# embeddings = model.encode(
#     df["clean_category"].tolist(),
#     normalize_embeddings=True,
#     show_progress_bar=True
# )

# embeddings = np.array(embeddings)



# bp_centroids = {}

# for bp in df["bp"].unique():

#     idx = df[df["bp"] == bp].index

#     centroid = embeddings[idx].mean(axis=0)

#     centroid = centroid / np.linalg.norm(centroid)

#     bp_centroids[bp] = centroid


# bp_list = list(bp_centroids.keys())

# centroid_matrix = np.vstack(list(bp_centroids.values()))



# similarity_matrix = cosine_similarity(embeddings, centroid_matrix)

# centroid_signal = []

# suggested_bp = []

# for i in range(len(df)):

#     current_bp = df.iloc[i]["bp"]

#     current_idx = bp_list.index(current_bp)

#     sim_current = similarity_matrix[i][current_idx]

#     best_idx = np.argmax(similarity_matrix[i])

#     best_bp = bp_list[best_idx]

#     sim_best = similarity_matrix[i][best_idx]

#     score = sim_best - sim_current

#     centroid_signal.append(score)

#     suggested_bp.append(best_bp)

# df["centroid_signal"] = centroid_signal
# df["centroid_suggested_bp"] = suggested_bp



# print("Computing KNN neighbors...")

# knn = NearestNeighbors(
#     n_neighbors=11,
#     metric="cosine"
# )

# knn.fit(embeddings)

# distances, indices = knn.kneighbors(embeddings)

# knn_signal = []

# for i in range(len(df)):

#     neighbor_ids = indices[i][1:]  # remove self

#     neighbor_bps = df.iloc[neighbor_ids]["bp"].values

#     current_bp = df.iloc[i]["bp"]

#     mismatch = np.sum(neighbor_bps != current_bp)

#     score = mismatch / len(neighbor_bps)

#     knn_signal.append(score)

# df["knn_signal"] = knn_signal



# print("Running HDBSCAN clustering...")

# clusterer = hdbscan.HDBSCAN(
#     min_cluster_size=15,
#     metric="euclidean"
# )

# clusters = clusterer.fit_predict(embeddings)

# df["cluster"] = clusters

# cluster_signal = []

# for i in range(len(df)):

#     cluster_id = df.iloc[i]["cluster"]

#     if cluster_id == -1:

#         cluster_signal.append(0)

#         continue

#     cluster_df = df[df["cluster"] == cluster_id]

#     majority_bp = cluster_df["bp"].mode()[0]

#     current_bp = df.iloc[i]["bp"]

#     if majority_bp != current_bp:

#         cluster_signal.append(1)

#     else:

#         cluster_signal.append(0)

# df["cluster_signal"] = cluster_signal



# distance_from_centroid = []

# for i in range(len(df)):

#     bp = df.iloc[i]["bp"]

#     centroid = bp_centroids[bp]

#     sim = cosine_similarity(
#         embeddings[i].reshape(1,-1),
#         centroid.reshape(1,-1)
#     )[0][0]

#     dist = 1 - sim

#     distance_from_centroid.append(dist)

# df["distance_to_bp"] = distance_from_centroid


# outlier_signal = []

# for bp in df["bp"].unique():

#     bp_rows = df[df["bp"] == bp]

#     mean_dist = bp_rows["distance_to_bp"].mean()

#     std_dist = bp_rows["distance_to_bp"].std()

#     for idx in bp_rows.index:

#         z = (df.loc[idx,"distance_to_bp"] - mean_dist) / (std_dist + 1e-9)

#         if z > 2:

#             outlier_signal.append((idx,1))

#         else:

#             outlier_signal.append((idx,0))


# outlier_dict = dict(outlier_signal)

# df["bp_outlier_signal"] = df.index.map(outlier_dict)



# df["final_score"] = (
#     0.40 * df["centroid_signal"] +
#     0.30 * df["knn_signal"] +
#     0.20 * df["cluster_signal"] +
#     0.10 * df["bp_outlier_signal"]
# )



# THRESHOLD = 0.35

# df["flag_misclassified"] = (
#     (df["final_score"] > THRESHOLD) &
#     (df["centroid_suggested_bp"] != df["bp"])
# )



# results = df[[
#     "category",
#     "bp",
#     "centroid_suggested_bp",
#     "final_score",
#     "flag_misclassified"
# ]].sort_values("final_score", ascending=False)


# results.to_csv("lineage_correction_results.csv", index=False)

# print("Pipeline completed.")
# print("Results saved to lineage_correction_results.csv")

# -------------------------------------------------------------------#


import re
import numpy as np
import pandas as pd
import spacy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import hdbscan



nlp = spacy.load("en_core_web_sm")



NEUTRAL_WORDS = {
    "for", "with", "on", "at", "to", "by", "of", "in", "upto", "and"
}

GENERIC_TERMS = {
    "service", "services",
    "center", "centers", "centre", "centres",
    "clinic", "clinics",
    "shop", "shops",
    "store", "stores",
    "class", "classes",
    "company", "companies",
    "hub", "hubs", "station",
    "cafe", "stations", "repair"
}



df = pd.read_excel("Sample_data.xlsx")

df = df.rename(columns={
    "jdmart_catname": "category",
    "BP": "bp"
})



def normalize_text(text: str) -> str:

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text



def linguistic_normalize(text: str):

    doc = nlp(text)

    tokens = []

    for token in doc:

        lemma = token.lemma_.lower()

        if lemma in NEUTRAL_WORDS or lemma in GENERIC_TERMS:
            continue

        if token.pos_ in {"NOUN", "ADJ", "PROPN"}:
            tokens.append(lemma)

    return " ".join(tokens)


df["clean_category"] = df["category"].apply(normalize_text)

df["normalized_text"] = df["clean_category"].apply(linguistic_normalize)


df["normalized_text"] = np.where(
    df["normalized_text"].str.len() > 0,
    df["normalized_text"],
    df["clean_category"]
)



print("Loading transformer model...")

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

print("Generating embeddings...")

embeddings = model.encode(
    df["normalized_text"].tolist(),
    normalize_embeddings=True,
    show_progress_bar=True
)

embeddings = np.array(embeddings)



bp_centroids = {}

for bp in df["bp"].unique():

    idx = df[df["bp"] == bp].index

    centroid = embeddings[idx].mean(axis=0)

    centroid = centroid / np.linalg.norm(centroid)

    bp_centroids[bp] = centroid


bp_list = list(bp_centroids.keys())

bp_index_map = {bp: i for i, bp in enumerate(bp_list)}

centroid_matrix = np.vstack(list(bp_centroids.values()))



similarity_matrix = cosine_similarity(embeddings, centroid_matrix)

centroid_signal = []
suggested_bp = []

for i in range(len(df)):

    current_bp = df.iloc[i]["bp"]

    current_idx = bp_index_map[current_bp]

    sim_current = similarity_matrix[i][current_idx]

    best_idx = np.argmax(similarity_matrix[i])

    best_bp = bp_list[best_idx]

    sim_best = similarity_matrix[i][best_idx]

    score = sim_best - sim_current

    centroid_signal.append(score)

    suggested_bp.append(best_bp)


df["centroid_signal"] = centroid_signal
df["centroid_suggested_bp"] = suggested_bp



print("Computing KNN neighbors...")

knn = NearestNeighbors(
    n_neighbors=11,
    metric="cosine"
)

knn.fit(embeddings)

distances, indices = knn.kneighbors(embeddings)

knn_signal = []

for i in range(len(df)):

    neighbor_ids = indices[i][1:]  # remove self

    neighbor_bps = df.iloc[neighbor_ids]["bp"].values

    current_bp = df.iloc[i]["bp"]

    mismatch = np.sum(neighbor_bps != current_bp)

    score = mismatch / len(neighbor_bps)

    knn_signal.append(score)


df["knn_signal"] = knn_signal



print("Running HDBSCAN clustering...")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=15,
    metric="euclidean"
)

clusters = clusterer.fit_predict(embeddings)

df["cluster"] = clusters

cluster_signal = []

for i in range(len(df)):

    cluster_id = df.iloc[i]["cluster"]

    if cluster_id == -1:
        cluster_signal.append(0)
        continue

    cluster_df = df[df["cluster"] == cluster_id]

    majority_bp = cluster_df["bp"].mode()[0]

    current_bp = df.iloc[i]["bp"]

    if majority_bp != current_bp:
        cluster_signal.append(1)
    else:
        cluster_signal.append(0)


df["cluster_signal"] = cluster_signal



distance_from_centroid = []

for i in range(len(df)):

    bp = df.iloc[i]["bp"]

    centroid = bp_centroids[bp]

    sim = cosine_similarity(
        embeddings[i].reshape(1, -1),
        centroid.reshape(1, -1)
    )[0][0]

    sim = np.clip(sim, -1, 1)

    dist = 1 - sim

    distance_from_centroid.append(dist)


df["distance_to_bp"] = distance_from_centroid

df["bp_outlier_signal"] = 0


for bp in df["bp"].unique():

    bp_rows = df[df["bp"] == bp]

    mean_dist = bp_rows["distance_to_bp"].mean()

    std_dist = bp_rows["distance_to_bp"].std()

    for idx in bp_rows.index:

        z = (df.loc[idx, "distance_to_bp"] - mean_dist) / (std_dist + 1e-9)

        if z > 2:
            df.loc[idx, "bp_outlier_signal"] = 1



df["final_score"] = (
    0.40 * df["centroid_signal"] +
    0.30 * df["knn_signal"] +
    0.20 * df["cluster_signal"] +
    0.10 * df["bp_outlier_signal"]
)



THRESHOLD = 0.35

df["flag_misclassified"] = (
    (df["final_score"] > THRESHOLD) &
    (df["centroid_suggested_bp"] != df["bp"])
)



results = df[[
    "category",
    "normalized_text",
    "clean_category",
    "bp",
    "centroid_signal",
    "knn_signal",
    "cluster_signal",
    "bp_outlier_signal",
    "final_score",
    "centroid_suggested_bp",
    "flag_misclassified"
]].sort_values("final_score", ascending=False)

results.to_csv("lineage_correction_results.csv", index=False)

print("Pipeline completed.")
print("Results saved to lineage_correction_results.csv")