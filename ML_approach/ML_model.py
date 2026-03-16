# =========================================================
# INSTALL REQUIRED LIBRARIES
# =========================================================
# pip install pandas scikit-learn sentence-transformers joblib

# =========================================================
# IMPORT LIBRARIES
# =========================================================
import re
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from sentence_transformers import SentenceTransformer
import joblib


print("===== CATEGORY LINEAGE TRAINING PIPELINE =====")


# =========================================================
# LOAD DATA
# =========================================================
# Dataset format assumed:
# category_name | base_parent

df = pd.read_excel("Sample_data.xlsx")

print("Dataset size:", df.shape)


# =========================================================
# TEXT CLEANING FUNCTION
# =========================================================

def clean_category(text):

    text = str(text).lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # remove numbers
    text = re.sub(r"\d+", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["clean_category"] = df["jdmart_catname"].apply(clean_category)


# =========================================================
# REMOVE DUPLICATES
# =========================================================

df = df.drop_duplicates(subset=["clean_category"])

print("After removing duplicates:", df.shape)


# =========================================================
# LABEL ENCODING BASE PARENTS
# =========================================================

label_encoder = LabelEncoder()

df["bp_encoded"] = label_encoder.fit_transform(df["BP"])

print("Number of Base Parents:", len(label_encoder.classes_))


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_category"],
    df["bp_encoded"],
    test_size=0.2,
    random_state=42,
    stratify=df["bp_encoded"]
)


print("Train size:", len(X_train))
print("Test size:", len(X_test))


# =========================================================
# LOAD SENTENCE TRANSFORMER
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

print("Encoding categories...")

X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)
X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)


# =========================================================
# TRAIN CLASSIFIER
# =========================================================

print("Training classifier...")

clf = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

clf.fit(X_train_emb, y_train)


# =========================================================
# PREDICTION
# =========================================================

y_pred = clf.predict(X_test_emb)

# probability for confidence
y_prob = clf.predict_proba(X_test_emb)
confidence = np.max(y_prob, axis=1)


# =========================================================
# CONVERT ENCODED LABELS BACK TO BP NAMES
# =========================================================

predicted_bp = label_encoder.inverse_transform(y_pred)
actual_bp = label_encoder.inverse_transform(y_test)


# =========================================================
# CREATE RESULT DATAFRAME
# =========================================================

results = pd.DataFrame({
    "category": X_test.values,
    "actual_bp": actual_bp,
    "predicted_bp": predicted_bp,
    "confidence": confidence
})


# =========================================================
# ADD CORRECT / INCORRECT FLAG
# =========================================================

results["correct_prediction"] = results["actual_bp"] == results["predicted_bp"]


# =========================================================
# SAVE RESULTS
# =========================================================

results.to_excel("category_bp_predictions.xlsx", index=False)

print("Prediction file saved: category_bp_predictions.xlsx")


# =========================================================
# EVALUATION
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(clf, "bp_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel saved successfully.")