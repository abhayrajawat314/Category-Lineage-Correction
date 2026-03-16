# =========================================================
# INSTALL
# =========================================================
# pip install transformers torch pandas scikit-learn openpyxl


# =========================================================
# IMPORT
# =========================================================
import pandas as pd
import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


print("===== STARTING TRAINING =====")


# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_excel("Sample_data.xlsx")
print("Rows:", len(df))


# =========================================================
# UNIQUE BASE PARENTS
# =========================================================
unique_bps = sorted(df["BP"].unique())
bp_to_idx = {bp: i for i, bp in enumerate(unique_bps)}

print("Total unique BPs:", len(unique_bps))


# =========================================================
# SPLIT
# =========================================================
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=42
)


# =========================================================
# LOAD MODEL
# =========================================================
model_name = "intfloat/e5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)


# =========================================================
# DATASET
# =========================================================
class CategoryDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.categories = df["jdmart_catname"].astype(str).tolist()
        self.labels = [bp_to_idx[bp] for bp in df["BP"]]

    def __len__(self):
        return len(self.categories)

    def __getitem__(self, idx):

        cat = "query: " + self.categories[idx]

        enc = tokenizer(
            cat,
            truncation=True,
            padding="max_length",
            max_length=12,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx])
        }


train_loader = DataLoader(
    CategoryDataset(train_df),
    batch_size=128,
    shuffle=True
)


# =========================================================
# PREPARE BP TEXTS
# =========================================================
bp_texts = ["passage: " + bp for bp in unique_bps]

bp_tokens = tokenizer(
    bp_texts,
    padding=True,
    truncation=True,
    max_length=12,
    return_tensors="pt"
)

bp_tokens = {k: v.to(device) for k, v in bp_tokens.items()}


# =========================================================
# OPTIMIZER
# =========================================================
optimizer = torch.optim.AdamW(
    encoder.parameters(),
    lr=5e-5
)


# =========================================================
# MEAN POOLING
# =========================================================
def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    summed = torch.sum(token_embeddings * mask, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)

    return summed / summed_mask


# =========================================================
# TRAINING LOOP
# =========================================================
print("===== TRAINING START =====")

encoder.train()

for epoch in range(3):

    print(f"\nEpoch {epoch+1}")
    total_loss = 0

    for step, batch in enumerate(train_loader):

        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)


        # CATEGORY EMBEDDINGS
        cat_out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        cat_emb = mean_pooling(cat_out, attention_mask)
        cat_emb = F.normalize(cat_emb, dim=1)


        # BP EMBEDDINGS
        bp_out = encoder(**bp_tokens)
        bp_emb = mean_pooling(bp_out, bp_tokens["attention_mask"])
        bp_emb = F.normalize(bp_emb, dim=1)


        # SIMILARITY MATRIX
        sim_matrix = torch.matmul(cat_emb, bp_emb.T)


        loss = F.cross_entropy(sim_matrix, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % 50 == 0:
            print(
                "Step", step,
                "Loss", round(loss.item(), 4),
                "AvgPosSim", round(sim_matrix[range(len(labels)), labels].mean().item(), 3)
            )

    print("Epoch loss:", total_loss)


print("Training complete")


# =========================================================
# SAVE MODEL
# =========================================================
encoder.save_pretrained("bp_embedding_model2")
tokenizer.save_pretrained("bp_embedding_model2")

print("Model saved.")