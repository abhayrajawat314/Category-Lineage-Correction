import numpy as np
from sentence_transformers import SentenceTransformer
from config import MODEL_NAME, EMBEDDING_PATH


def generate_embeddings(texts):

    print("Loading embedding model")
    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings")

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings)

    np.save(EMBEDDING_PATH, embeddings)

    print("Embeddings saved:", EMBEDDING_PATH)

    return embeddings