import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier

from config import FEATURE_PATH, META_MODEL_PATH


FEATURES = [
    "sim_current_bp",
    "sim_best_bp",
    "similarity_margin",
    "knn_mismatch_ratio",
    "knn_entropy",
    "cluster_consistency",
    "bp_outlier"
]


print("Loading training features")

df = pd.read_csv(FEATURE_PATH)

X = df[FEATURES]

y = df["bp"]


X_train,X_test,y_train,y_test = train_test_split(
    X,y,
    test_size=0.2,
    random_state=42
)


print("Training model")

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42
)

model.fit(X_train,y_train)


preds = model.predict(X_test)

print(classification_report(y_test,preds))


joblib.dump(model,META_MODEL_PATH)

print("Model saved:", META_MODEL_PATH)