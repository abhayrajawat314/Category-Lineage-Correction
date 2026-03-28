import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

from config import FEATURE_PATH, META_MODEL_PATH,LABEL_ENCODER_PATH,EMBEDDING_PATH
import numpy as np



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


print("Loading training features")

df = pd.read_csv(FEATURE_PATH)

print("Loading embeddings")

embeddings = np.load(EMBEDDING_PATH)

embedding_df = pd.DataFrame(embeddings)

X = pd.concat([df[FEATURES], embedding_df], axis=1)

# ---------------------------------------
# LABEL ENCODING
# ---------------------------------------

print("Encoding BP labels")

le = joblib.load(LABEL_ENCODER_PATH)
y = df["bp_encoded"]



# ---------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ---------------------------------------
# MODEL
# ---------------------------------------

print("Training XGBoost model with progress...")

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=len(le.classes_),
    random_state=42
)


# ---------------------------------------
# TRAINING (FIXED)
# ---------------------------------------

model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50   # this WILL work across versions
)


# ---------------------------------------
# EVALUATION
# ---------------------------------------

preds = model.predict(X_test)

y_test_labels = le.inverse_transform(y_test)
pred_labels = le.inverse_transform(preds)

print("\nModel Evaluation")
print(classification_report(y_test_labels, pred_labels))


# ---------------------------------------
# SAVE MODEL
# ---------------------------------------

joblib.dump(model, META_MODEL_PATH)

print("Model saved:", META_MODEL_PATH)