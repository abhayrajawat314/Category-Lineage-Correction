import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from config import KNN_NEIGHBORS, CLUSTER_MIN_SIZE


def compute_knn_signal(df, embeddings):

    print("Computing KNN signals")

    knn = NearestNeighbors(
        n_neighbors=KNN_NEIGHBORS + 1,
        metric="cosine"
    )

    knn.fit(embeddings)

    distances, indices = knn.kneighbors(embeddings)

    mismatch_ratio = []
    entropy_list = []

    for i in range(len(df)):

        neighbor_ids = indices[i][1:]

        neighbor_bps = df.iloc[neighbor_ids]["bp"].values

        current_bp = df.iloc[i]["bp"]

        mismatch = np.sum(neighbor_bps != current_bp)

        ratio = mismatch / len(neighbor_bps)

        mismatch_ratio.append(ratio)

        counts = pd.Series(neighbor_bps).value_counts(normalize=True)

        entropy = -(counts * np.log(counts)).sum()

        entropy_list.append(entropy)

    df["knn_mismatch_ratio"] = mismatch_ratio
    df["knn_entropy"] = entropy_list

    return df


def compute_cluster_signal(df, embeddings):

    print("Running clustering")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=CLUSTER_MIN_SIZE,
        metric="euclidean"
    )

    clusters = clusterer.fit_predict(embeddings)

    df["cluster_id"] = clusters

    cluster_consistency = []

    cluster_groups = df.groupby("cluster_id")

    cluster_bp_dist = {}

    for cid, group in cluster_groups:

        counts = group["bp"].value_counts(normalize=True)

        cluster_bp_dist[cid] = counts

    for i in range(len(df)):

        cid = df.iloc[i]["cluster_id"]
        bp = df.iloc[i]["bp"]

        if cid == -1:
            cluster_consistency.append(0)
            continue

        dist = cluster_bp_dist[cid]

        consistency = dist.get(bp, 0)

        cluster_consistency.append(consistency)

    df["cluster_consistency"] = cluster_consistency

    return df


def compute_centroid_signals(df, embeddings, bp_centroids):

    print("Computing centroid signals")

    bp_list = list(bp_centroids.keys())

    centroid_matrix = np.vstack(list(bp_centroids.values()))

    sim_matrix = cosine_similarity(embeddings, centroid_matrix)

    sim_current = []
    sim_best = []
    sim_second = []
    best_bp_list = []

    for i in range(len(df)):

        sims = sim_matrix[i]

        current_bp = df.iloc[i]["bp"]
        current_idx = bp_list.index(current_bp)

        sim_current.append(sims[current_idx])

        sorted_idx = np.argsort(sims)[::-1]

        best_idx = sorted_idx[0]
        second_idx = sorted_idx[1]

        sim_best.append(sims[best_idx])
        sim_second.append(sims[second_idx])

        best_bp_list.append(bp_list[best_idx])

    df["sim_current_bp"] = sim_current
    df["sim_best_bp"] = sim_best
    df["sim_second_bp"] = sim_second

    df["similarity_margin"] = df["sim_best_bp"] - df["sim_second_bp"]

    df["centroid_suggested_bp"] = best_bp_list

    return df


def compute_bp_outlier(df, embeddings):

    print("Computing BP outliers using embedding distribution")

    df["bp_outlier"] = 0

    for bp in df["bp"].unique():

        idx = df[df["bp"] == bp].index

        bp_embeddings = embeddings[idx]

        mean = bp_embeddings.mean(axis=0)
        std = bp_embeddings.std(axis=0) + 1e-9

        z_scores = np.linalg.norm((bp_embeddings - mean) / std, axis=1)

        threshold = np.percentile(z_scores, 95)

        outliers = z_scores > threshold

        df.loc[idx[outliers], "bp_outlier"] = 1

    return df