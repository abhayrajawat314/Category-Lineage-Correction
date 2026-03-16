from torch.utils.data import DataLoader
from sentence_transformers import losses

from config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_SAVE_PATH
)

from data_loader import load_dataset
from pair_generation import generate_training_pairs
from model_utils import build_model


print("Loading dataset")

df = load_dataset()


print("Generating pairs")

train_examples = generate_training_pairs(df)


train_loader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE
)


print("Building model")

model = build_model()


train_loss = losses.MultipleNegativesRankingLoss(model)


print("Starting training")

model.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=500,
    optimizer_params={"lr": LEARNING_RATE},
    show_progress_bar=True
)


print("Saving model")

model.save(MODEL_SAVE_PATH)

print("Training complete")