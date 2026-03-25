import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------- Data parameters ----------------------
MODEL = 'Starcoder_768'
MERGE_TYPE = 'concatenate'
DATA_DIR = f"./merged_features/{MERGE_TYPE}/{MODEL}/"
LABEL_PATH = "./dataset/label.csv"
HAS_HEADER = True

# Specify the list of positive samples that must be included in the test set (validation set).
TEST_POSITIVE_PROTOCOLS = {
    "LAURAToken", "RoulettePotV2", "Paribus", "Ast", "Peapods Finance",
    "SBR Token", "H2O", "DCFToken", "BBXToken", "ImpermaxV3(1)", "ImpermaxV3(2)",
    "Lifeprotocol", "Nalakuvara_LotteryTicket", "MBUToken",
    "GradientMakerPool", "ResupplyFi", "GMX", "YuliAI", "PDZ",
    "d3xai", "Balancer", "DRLVaultV3"
}


def get_embed_dim(data_dir):
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file was found in the data directory {data_dir}.")
    first_csv_path = os.path.join(data_dir, csv_files[0])
    df = pd.read_csv(first_csv_path, header=0 if HAS_HEADER else None)
    return df.shape[1]


EMBED_DIM = get_embed_dim(DATA_DIR)

# ---------------------- Model parameters ----------------------
NUM_HEADS = 8
assert EMBED_DIM % NUM_HEADS == 0
NUM_ENCODERS = 2
HIDDEN_DIM = 768
DROPOUT = 0.3

# ---------------------- Training parameters ----------------------
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
MODEL_SAVE_PATH = "defi_manipulation_model.pth"
POS_NEG_RATIO = 0.5
PRED_THRESHOLD = 0.1
PATIENCE = 3


class DeFiEmbeddingDataset(Dataset):
    def __init__(self, data_dir, label_dict, embed_dim, has_header=False):
        self.data_dir = data_dir
        self.label_dict = label_dict
        self.embed_dim = embed_dim
        self.has_header = has_header
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        self.seq_lens = self._precompute_seq_lens()

    def _precompute_seq_lens(self):
        seq_lens = {}
        for file_name in self.file_names:
            file_path = os.path.join(self.data_dir, file_name)
            df = pd.read_csv(file_path, header=0 if self.has_header else None)
            seq_lens[file_name] = df.shape[0]
        return seq_lens

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path, header=0 if self.has_header else None)
        embedding = torch.tensor(df.values, dtype=torch.float32)
        protocol_name = file_name[:-4]
        label = torch.tensor([self.label_dict[protocol_name]], dtype=torch.float32)
        return embedding, label, protocol_name  # The returned protocol name is used for tracking TP.


def load_label_dict(label_path):
    label_dict = {}
    with open(label_path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                file_name, label = parts[0], parts[1]
                label_dict[file_name] = int(label)
    return label_dict


def collate_fn(batch):
    embeddings, labels, names = zip(*batch)
    max_seq_len = max(emb.shape[0] for emb in embeddings)
    padded_embeddings = []
    for emb in embeddings:
        pad_length = max_seq_len - emb.shape[0]
        padded = torch.cat([emb, torch.zeros(pad_length, EMBED_DIM, dtype=torch.float32)],
                           dim=0) if pad_length > 0 else emb
        padded_embeddings.append(padded)

    batch_emb = torch.stack(padded_embeddings, dim=0)
    batch_labels = torch.stack(labels, dim=0)
    attention_mask = torch.zeros(batch_emb.shape[:2], dtype=torch.float32)
    for i, emb in enumerate(embeddings):
        attention_mask[i, :emb.shape[0]] = 1.0
    return batch_emb, batch_labels, attention_mask, names


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=2500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_relative_pos=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_pos = max_relative_pos
        self.relative_emb = nn.Embedding(2 * max_relative_pos + 1, embed_dim)
        self.register_buffer('relative_range', torch.arange(-max_relative_pos, max_relative_pos + 1, dtype=torch.long))

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        device = x.device
        i = torch.arange(seq_len, device=device).unsqueeze(1)
        relative_pos = i - (i + self.relative_range.unsqueeze(0))
        relative_pos_clamped = torch.clamp(relative_pos, -self.max_relative_pos, self.max_relative_pos)
        relative_pos_ids = relative_pos_clamped + self.max_relative_pos
        relative_emb = self.relative_emb(relative_pos_ids)
        relative_emb_avg = relative_emb.mean(dim=1)
        x = x + relative_emb_avg.unsqueeze(0)
        return self.dropout(x)


class DeFiTransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_encoders, hidden_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len=2500, dropout=dropout)
        self.relative_pos_encoding = RelativePositionalEncoding(embed_dim=embed_dim, max_relative_pos=50,
                                                                dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x, attention_mask=None):
        x = self.relative_pos_encoding(x)
        batch_size, seq_len, _ = x.shape
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask) * -10000.0
            mask = mask.unsqueeze(1).repeat(1, seq_len, 1)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask = mask.view(batch_size * self.num_heads, seq_len, seq_len)
        x = self.transformer_encoder(x, mask=mask)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            x = x.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, device, threshold=PRED_THRESHOLD):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in loader:
        embeddings, labels, attention_mask, _ = batch
        embeddings, labels, attention_mask = embeddings.to(device), labels.to(device), attention_mask.to(device)
        preds = model(embeddings, attention_mask)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * embeddings.size(0)
        probs = torch.sigmoid(preds).detach().cpu().numpy()
        all_preds.extend((probs >= threshold).astype(int))
        all_labels.extend(labels.detach().cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds,
                                                                                             zero_division=0), precision_score(
        all_labels, all_preds, zero_division=0), recall_score(all_labels, all_preds, zero_division=0)


def validate(model, loader, criterion, device, threshold=PRED_THRESHOLD):
    model.eval()
    total_loss = 0.0
    all_preds, all_probs, all_labels, all_names = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            embeddings, labels, attention_mask, names = batch
            embeddings, labels, attention_mask = embeddings.to(device), labels.to(device), attention_mask.to(device)
            preds = model(embeddings, attention_mask)
            total_loss += criterion(preds, labels).item() * embeddings.size(0)
            probs = torch.sigmoid(preds).detach().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= threshold).astype(int))
            all_labels.extend(labels.detach().cpu().numpy())
            all_names.extend(names)

    avg_loss = total_loss / len(loader.dataset)
    all_preds, all_labels = np.array(all_preds).flatten(), np.array(all_labels).flatten()

    # Identify the specific TP protocol.
    tp_protocols = [all_names[i] for i in range(len(all_names)) if all_preds[i] == 1 and all_labels[i] == 1]

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    tp, tn = ((all_preds == 1) & (all_labels == 1)).sum(), ((all_preds == 0) & (all_labels == 0)).sum()
    fp, fn = ((all_preds == 1) & (all_labels == 0)).sum(), ((all_preds == 0) & (all_labels == 1)).sum()

    return avg_loss, acc, f1, prec, rec, auc, tp, tn, fp, fn, tp_protocols


def run_experiment(exp_id):
    label_dict = load_label_dict(LABEL_PATH)
    full_dataset = DeFiEmbeddingDataset(DATA_DIR, label_dict, EMBED_DIM, has_header=HAS_HEADER)

    train_indices, val_indices = [], []
    neg_indices = []

    for i, file_name in enumerate(full_dataset.file_names):
        protocol = file_name[:-4]
        label = label_dict[protocol]
        if label == 1:
            if protocol in TEST_POSITIVE_PROTOCOLS:
                val_indices.append(i)
            else:
                train_indices.append(i)
        else:
            neg_indices.append(i)

    # Negative samples are divided proportionally.
    random.shuffle(neg_indices)
    split_point = int(len(neg_indices) * (1 - VAL_SPLIT))
    train_indices.extend(neg_indices[:split_point])
    val_indices.extend(neg_indices[split_point:])

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = DeFiTransformerClassifier(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, num_encoders=NUM_ENCODERS,
                                      hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)

    # 计算权重
    train_labels = [label_dict[full_dataset.file_names[i][:-4]] for i in train_indices]
    pos_c, neg_c = sum(train_labels), len(train_labels) - sum(train_labels)
    pos_weight = torch.tensor(neg_c / pos_c if pos_c > 0 else 1.0, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)

    best_val_f1, best_metrics, best_tps = 0.0, {}, []
    early_stop_counter = 0

    print(
        f"\nExperiment {exp_id} started | Training set positive samples: {pos_c}, Validation set positive samples: {len([i for i in val_indices if label_dict[full_dataset.file_names[i][:-4]] == 1])}")

    for epoch in range(EPOCHS):
        t_loss, t_acc, t_f1, t_prec, t_rec = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, v_f1, v_prec, v_rec, v_auc, tp, tn, fp, fn, tp_list = validate(model, val_loader, criterion,
                                                                                      device)
        scheduler.step(v_loss)

        # Print epoch information
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Training Set - Loss: {t_loss:.4f}, Accuracy: {t_acc:.4f}, F1: {t_f1:.4f}")
        print(f"          Precision: {t_prec:.4f}, Recall: {t_rec:.4f}")
        print(f"Validation Set - Loss: {v_loss:.4f}, Accuracy: {v_acc:.4f}, F1: {v_f1:.4f}, AUC: {v_auc:.4f}")
        print(f"          Precision: {v_prec:.4f}, Recall: {v_rec:.4f}")
        print(f"          TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_metrics = {'acc': v_acc, 'precision': v_prec, 'recall': v_rec, 'f1': v_f1, 'auc': v_auc}
            best_tps = tp_list
            torch.save(model.state_dict(), f"{exp_id}_{MODEL_SAVE_PATH}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"Early stop count: {early_stop_counter}/{PATIENCE}")
            if early_stop_counter >= PATIENCE:
                print(f"The early stopping mechanism was triggered, and training stopped at epoch {epoch + 1}.")
                break

    print(f"Experiment {exp_id}: Best TP protocols ({len(best_tps)} protocols): {', '.join(best_tps)}")
    return best_metrics


if __name__ == "__main__":
    all_results = []
    train_time = 0

    for i in range(10):
        start = time.time()
        metrics = run_experiment(i)
        end = time.time()
        train_time += end - start

        all_results.append(metrics)

    df_results = pd.DataFrame(all_results)
    print("\n" + "=" * 30 + " statistical indicators " + "=" * 30)
    for col in df_results.columns:
        print(f"{col:10}: {df_results[col].mean():.4f} ± {df_results[col].std():.4f}")

    print(train_time)