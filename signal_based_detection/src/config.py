MODEL_NAME = "BAAI/bge-large-en-v1.5"

DATA_PATH = "data/top_3000_result1 (1).xlsx"

EMBEDDING_BATCH_SIZE = 256

KNN_NEIGHBORS = 20
CLUSTER_MIN_SIZE = 15

ARTIFACT_DIR = "signal_based_detection/artifacts/"

EMBEDDING_PATH = ARTIFACT_DIR + "category_embeddings1.npy"
CENTROID_PATH = ARTIFACT_DIR + "bp_centroids1.npy"
BP_LABEL_PATH = ARTIFACT_DIR + "bp_labels1.npy"

FEATURE_PATH = ARTIFACT_DIR + "signal_features2.csv"

META_MODEL_PATH = ARTIFACT_DIR + "meta_model3.pkl"
SCALER_PATH = ARTIFACT_DIR + "signal_scaler1.pkl"

RESULT_PATH = ARTIFACT_DIR + "lineage_results3.csv"
FINAL_RESULT = ARTIFACT_DIR + "signal_results4.csv"

LABEL_ENCODER_PATH=ARTIFACT_DIR + "label_encoder2.pkl"

TEST_DATA_PATH="data/Testinf_data.xlsx"

TEST_RESULT=ARTIFACT_DIR + "result1.csv"