import random
import pandas as pd
from sentence_transformers import InputExample
from config import TRAINING_FILE_PATH


def generate_training_pairs(df):

    print("Generating training pairs")

    positive_pairs = []

    grouped = df.groupby("bp")

    for bp, group in grouped:

        categories = group["category"].tolist()

        if len(categories) < 2:
            continue

        for i in range(len(categories) - 1):

            anchor = categories[i]

            positive = random.choice(categories)

            # avoid identical pairs
            if anchor == positive:
                continue

            positive_pairs.append(
                InputExample(
                    texts=[
                        "query: " + anchor,
                        "passage: " + positive
                    ]
                )
            )

    print("Total positive pairs:", len(positive_pairs))

    # ---------------------------------------
    # SAVE PAIRS FOR INSPECTION
    # ---------------------------------------

    pairs_df = pd.DataFrame({
        "query": [p.texts[0] for p in positive_pairs],
        "positive": [p.texts[1] for p in positive_pairs]
    })

    pairs_df.to_csv(
        TRAINING_FILE_PATH,
        index=False
    )

    print("Pairs saved")

    return positive_pairs