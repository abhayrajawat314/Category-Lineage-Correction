from sentence_transformers import SentenceTransformer, models
from config import BASE_MODEL, MAX_SEQ_LENGTH


def build_model():

    transformer = models.Transformer(
        BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH
    )

    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True
    )

    model = SentenceTransformer(modules=[transformer, pooling])

    return model