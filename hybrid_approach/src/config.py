# =============================
# BASE MODEL
# =============================
BASE_MODEL = "intfloat/e5-base"

# =============================
# PATHS
# =============================
DATA_PATH = "data/top_3000_result1 (1).xlsx"
TEST_DATA_PATH = "data/Testinf_data.xlsx"

ARTIFACT_DIR = "hybrid_approach/artifacts/"

# =============================
# FINETUNED MODEL
# =============================
FINETUNED_MODEL_PATH = ARTIFACT_DIR + "finetuned_model1"

# =============================
# TRAINING PARAMS
# =============================
MAX_SEQ_LENGTH = 12
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 5e-5
TRAIN_SAMPLE_SIZE = 100000

TRAIN_PATH = ARTIFACT_DIR + "train1.csv"
TEST_PATH = ARTIFACT_DIR + "test1.csv"
TRAINING_FILE_PATH = ARTIFACT_DIR + "training_pairs1.csv"

# =============================
# SIGNAL PIPELINE
# =============================
EMBEDDING_PATH = ARTIFACT_DIR + "embeddings1.npy"
CENTROID_PATH = ARTIFACT_DIR + "centroids1.npy"
BP_LABEL_PATH = ARTIFACT_DIR + "bp_labels1.npy"

FEATURE_PATH = ARTIFACT_DIR + "features1.csv"

META_MODEL_PATH = ARTIFACT_DIR + "meta_model1.pkl"
SCALER_PATH = ARTIFACT_DIR + "scaler1.pkl"
LABEL_ENCODER_PATH = ARTIFACT_DIR + "label_encoder1.pkl"

RESULT_PATH = ARTIFACT_DIR + "final_results1.csv"

# =============================
# SIGNAL PARAMS
# =============================
KNN_NEIGHBORS = 20
CLUSTER_MIN_SIZE = 15