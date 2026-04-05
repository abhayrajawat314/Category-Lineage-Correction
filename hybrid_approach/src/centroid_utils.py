import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from config import CENTROID_PATH, BP_LABEL_PATH


def compute_bp_centroids(df, embeddings):

    bp_centroids = {}

    for bp in df["bp"].unique():

        idx = df[df["bp"] == bp].index

        centroid = embeddings[idx].mean(axis=0)

        centroid = centroid / np.linalg.norm(centroid)

        bp_centroids[bp] = centroid

    centroid_matrix = np.vstack(list(bp_centroids.values()))
    bp_list = list(bp_centroids.keys())

    np.save(CENTROID_PATH, centroid_matrix)
    np.save(BP_LABEL_PATH, np.array(bp_list))

    print("Centroids saved")

    return bp_centroids


def centroid_similarity_matrix(embeddings, bp_centroids):

    centroid_matrix = np.vstack(list(bp_centroids.values()))
    bp_list = list(bp_centroids.keys())

    sim_matrix = cosine_similarity(embeddings, centroid_matrix)

    return sim_matrix, bp_list