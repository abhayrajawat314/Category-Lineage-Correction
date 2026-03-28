MODEL_NAME = "BAAI/bge-large-en-v1.5"

DATA_PATH = "data/top_3000_result1 (1).xlsx"

EMBEDDING_BATCH_SIZE = 256

KNN_NEIGHBORS = 20
CLUSTER_MIN_SIZE = 15

ARTIFACT_DIR = "signal_based_detection/artifacts/"

EMBEDDING_PATH = ARTIFACT_DIR + "category_embeddings.npy"
CENTROID_PATH = ARTIFACT_DIR + "bp_centroids.npy"
BP_LABEL_PATH = ARTIFACT_DIR + "bp_labels.npy"

FEATURE_PATH = ARTIFACT_DIR + "signal_features1.csv"

META_MODEL_PATH = ARTIFACT_DIR + "meta_model2.pkl"
SCALER_PATH = ARTIFACT_DIR + "signal_scaler.pkl"

RESULT_PATH = ARTIFACT_DIR + "lineage_results2.csv"
FINAL_RESULT = ARTIFACT_DIR + "signal_results3.csv"

LABEL_ENCODER_PATH=ARTIFACT_DIR + "label_encoder.pkl"

TEST_DATA_PATH="data/Testinf_data.xlsx"

TEST_RESULT=ARTIFACT_DIR + "result.csv"