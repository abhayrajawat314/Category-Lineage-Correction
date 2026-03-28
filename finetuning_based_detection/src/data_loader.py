import pandas as pd
import os
from transformers import AutoTokenizer
from config import DATA_PATH, TRAIN_SAMPLE_SIZE,TRAIN_PATH,TEST_PATH




def load_dataset():

    # ===============================
    # LOAD EXISTING SPLIT
    # ===============================
    if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):

        print("Loading existing train/test split")

        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)

        return train_df, test_df

    # ===============================
    # CREATE NEW SPLIT
    # ===============================
    print("Creating new train/test split")

    df = pd.read_excel(DATA_PATH)

    df = df.rename(columns={
        "jdmart_catname": "category",
        "BP": "bp"
    })

    df["category"] = df["category"].astype(str)

    if len(df) > TRAIN_SAMPLE_SIZE:
        train_df = df.sample(TRAIN_SAMPLE_SIZE, random_state=42)
        test_df = df.drop(train_df.index).reset_index(drop=True)
    else:
        train_df = df.copy()
        test_df = pd.DataFrame(columns=df.columns)

    print("Train size:", len(train_df))
    print("Test size:", len(test_df))

    # ===============================
    # SAVE SPLIT (CRITICAL)
    # ===============================
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("Train/Test split saved")

    # ===============================
    # TOKEN CHECK (optional)
    # ===============================
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")

    token_lengths = train_df["category"].apply(
        lambda x: len(tokenizer.tokenize(x))
    )

    print("Max tokens:", token_lengths.max())
    print("Avg tokens:", token_lengths.mean())

    return train_df, test_df