from sentence_transformers import SentenceTransformer, models
from config import MODEL_NAME, MAX_SEQ_LENGTH


def build_model():

    transformer = models.Transformer(
        MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH
    )

    # mean pooling
    pooling = models.Pooling(
        transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
    )

    model = SentenceTransformer(modules=[transformer, pooling])

    return model