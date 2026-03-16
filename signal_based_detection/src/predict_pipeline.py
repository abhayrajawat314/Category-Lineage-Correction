import pandas as pd
import joblib

from config import FEATURE_PATH, META_MODEL_PATH, FINAL_RESULT


FEATURES = [
    "sim_current_bp",
    "sim_best_bp",
    "similarity_margin",
    "knn_mismatch_ratio",
    "knn_entropy",
    "cluster_consistency",
    "bp_outlier"
]


print("Loading features")

df = pd.read_csv(FEATURE_PATH)


print("Loading trained model")

model = joblib.load(META_MODEL_PATH)


print("Predicting BP")

df["predicted_bp"] = model.predict(df[FEATURES])


df.to_csv(FINAL_RESULT, index=False)

print("Prediction results saved:", FINAL_RESULT)