MODEL_NAME = "intfloat/e5-base"

DATA_PATH = "data/top_3000_result1 (1).xlsx"

ARTIFACT_DIR="finetuning_based_detection/artifacts/"

MODEL_SAVE_PATH = ARTIFACT_DIR+"trained_bp_embedding_model2"

MAX_SEQ_LENGTH = 12

BATCH_SIZE = 64

EPOCHS = 3

LEARNING_RATE = 5e-5

TRAIN_SAMPLE_SIZE = 60000

TRAINING_FILE_PATH=ARTIFACT_DIR+"training_pairs2.csv"

RESULT_PATH="finetuning_based_detection/artifacts/metric_learning_results3.csv"

EMBEDDING_PATH=ARTIFACT_DIR+"category_embeddings2.npy"

CENTROID_PATH=ARTIFACT_DIR+"bp_centroids2.npy"

BP_LABEL_PATH=ARTIFACT_DIR+"bp_labels2.npy"

TRAIN_PATH=ARTIFACT_DIR+"train.csv"

TEST_PATH=ARTIFACT_DIR+"test.csv"

TEST_RESULT=ARTIFACT_DIR + "result.csv"

TEST_DATA_PATH="data/Testinf_data.xlsx"