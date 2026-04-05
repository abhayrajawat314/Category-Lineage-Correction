import numpy as np
from sentence_transformers import SentenceTransformer
from config import FINETUNED_MODEL_PATH, EMBEDDING_PATH


def generate_embeddings(texts):

    print("Loading FINETUNED model")

    model = SentenceTransformer(FINETUNED_MODEL_PATH)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)

    np.save(EMBEDDING_PATH, embeddings)

    print("Saved embeddings")

    return embeddings