import pandas as pd
from config import DATA_PATH, TRAIN_SAMPLE_SIZE
from transformers import AutoTokenizer


def load_dataset():

    df = pd.read_excel(DATA_PATH)

    df = df.rename(columns={
        "jdmart_catname": "category",
        "BP": "bp"
    })

    df["category"] = df["category"].astype(str)

    if len(df) > TRAIN_SAMPLE_SIZE:
        df = df.sample(TRAIN_SAMPLE_SIZE, random_state=42)


    print("Checking token lengths")

    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-base")

    token_lengths = df["category"].apply(
        lambda x: len(tokenizer.tokenize(x))
    )

    print("Max tokens in categories:", token_lengths.max())
    print("Average tokens:", token_lengths.mean())

    return df