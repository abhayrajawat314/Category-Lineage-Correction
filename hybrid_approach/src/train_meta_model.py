import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from config import FEATURE_PATH, META_MODEL_PATH, EMBEDDING_PATH, LABEL_ENCODER_PATH


FEATURES = [
    "sim_best_bp",
    "sim_second_bp",
    "similarity_margin",
    "knn_mismatch_ratio",
    "knn_entropy",
    "cluster_consistency",
    "bp_outlier",
    "current_bp_rank"
]

print("Loading features")

df = pd.read_csv(FEATURE_PATH)
embeddings = np.load(EMBEDDING_PATH)

X = pd.concat([df[FEATURES], pd.DataFrame(embeddings)], axis=1)

le = joblib.load(LABEL_ENCODER_PATH)
y = df["bp_encoded"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, stratify=y
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=len(le.classes_)
)

model.fit(X_train, y_train)

joblib.dump(model, META_MODEL_PATH)

print("Meta model trained")