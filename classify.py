import os
import random
import time
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
print(f"Using the device: {device}")

# ---------------------- Data parameters ----------------------
MODEL = 'Starcoder_768' # Starcoder_768, Codegemma_768, deepseek_768, Qwen_768, SmolLM2_768
MERGE_TYPE = 'concatenate' # 'concatenate', 'weighted_concat'
DATA_DIR = F"./merged_features/{MERGE_TYPE}/{MODEL}/"
LABEL_PATH = "./dataset/label.csv"
HAS_HEADER = True


# Dynamically obtain EMBED_DIM: Read the number of columns in the first CSV file as the embedding dimension.
def get_embed_dim(data_dir):
    """
    Automatically obtain the embedding dimension from the data directory.
    Read the number of columns in the first CSV file and verify that all files have the same number of columns.
    """
    # Get all CSV files in the directory.
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file was found in the data directory {data_dir}. Please check if the path is correct.")

    # Read the first file to get the number of columns.
    first_csv_path = os.path.join(data_dir, csv_files[0])
    df = pd.read_csv(first_csv_path, header=0 if HAS_HEADER else None)
    embed_dim = df.shape[1]
    print(f"Automatically retrieving EMBED_DIM from the CSV file: {embed_dim}")

    # Verify that all CSV files have the same number of columns to avoid dimension mismatches.
    for csv_file in csv_files[1:]:
        csv_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(csv_path, header=0 if HAS_HEADER else None)
        assert df.shape[1] == embed_dim, \
            f"The number of columns in the file {csv_file} is inconsistent! It should be {embed_dim} columns, but it actually has {df.shape[1]} columns."
    return embed_dim


EMBED_DIM = get_embed_dim(DATA_DIR)

# ---------------------- Model parameters ----------------------
NUM_HEADS = 8
assert EMBED_DIM % NUM_HEADS == 0, \
    f"EMBED_DIM ({EMBED_DIM}) must be divisible by NUM_HEADS ({NUM_HEADS}). Please adjust the value of NUM_HEADS."
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
PATIENCE = 3  # Patience value of the early stopping mechanism


class DeFiEmbeddingDataset(Dataset):
    def __init__(self, data_dir, label_dict, embed_dim, has_header=False):
        """
        Dataset Class: Loads embedding files and corresponding labels
        :param data_dir: Directory containing embedding files
        :param label_dict: Label dictionary (filename -> label)
        :param embed_dim: Dynamically obtained embedding dimension
        :param has_header: Whether the CSV file has a header
        """
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
            seq_len = df.shape[0]
            # Verify that the number of columns matches the dynamic dimensions.
            assert df.shape[1] == self.embed_dim, \
                f"CSV column count error! {file_name} should have {self.embed_dim} columns, but actually has {df.shape[1]} columns."
            seq_lens[file_name] = seq_len
        return seq_lens

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        """Obtain a single sample: embedding + label"""
        file_name = self.file_names[idx]
        file_path = os.path.join(self.data_dir, file_name)
        df = pd.read_csv(file_path, header=0 if self.has_header else None)
        embedding_np = df.values
        seq_len = embedding_np.shape[0]
        # Verify that the sequence length matches the pre-calculated value.
        assert seq_len == self.seq_lens[file_name], \
            f"File {file_name} length has changed! Pre-calculated length: {self.seq_lens[file_name]}, actual length: {seq_len}"
        embedding = torch.tensor(embedding_np, dtype=torch.float32)
        label = torch.tensor([self.label_dict[file_name[:-4]]], dtype=torch.float32)
        return embedding, label


def load_label_dict(label_path):
    """Loading the label dictionary: Reading filenames and corresponding labels from a CSV file."""
    label_dict = {}
    with open(label_path, "r", encoding="utf-8") as f:
        next(f) 
        for line in f:
            file_name, label = line.strip().split(",")
            label_dict[file_name] = int(label)
    return label_dict


def collate_fn(batch):
    """
    Custom batch processing function: Handles padding and attention masks for variable-length sequences.
    :param batch: Batch samples（list of (embedding, label)）
    :return: padded_embeddings, batch_labels, attention_mask
    """
    embeddings, labels = zip(*batch)
    max_seq_len = max(emb.shape[0] for emb in embeddings)
    padded_embeddings = []
    for emb in embeddings:
        pad_length = max_seq_len - emb.shape[0]
        padded = torch.cat([emb, torch.zeros(pad_length, EMBED_DIM, dtype=torch.float32)],
                           dim=0) if pad_length > 0 else emb
        padded_embeddings.append(padded)
    batch_emb = torch.stack(padded_embeddings, dim=0)
    batch_labels = torch.stack(labels, dim=0)
    # Generating attention masks: 1 indicates a valid position, 0 indicates a padding position.
    attention_mask = torch.zeros(batch_emb.shape[:2], dtype=torch.float32)
    for i, emb in enumerate(embeddings):
        attention_mask[i, :emb.shape[0]] = 1.0
    return batch_emb, batch_labels, attention_mask


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=2500, dropout=0.1):
        """
        Positional Encoding: Adds positional information to the sequence, enabling the Transformer to capture sequential features.
        :param embed_dim: Embedding dimension (dynamically obtained)
        :param max_seq_len: Maximum sequence length
        :param dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-np.log(10000.0) / embed_dim))
        pe = torch.zeros(max_seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding and return the result."""
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)



# Relative positional encoding
class RelativePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_relative_pos=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.max_relative_pos = max_relative_pos
        self.relative_emb = nn.Embedding(2 * max_relative_pos + 1, embed_dim)

        # Pre-calculate the relative position range (-max_relative_pos to max_relative_pos)
        self.register_buffer(
            'relative_range',
            torch.arange(-max_relative_pos, max_relative_pos + 1, dtype=torch.long)
        )
        self.range_size = self.relative_range.size(0)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        device = x.device

        # Generate relative position indices
        # Directly generate the relative position range [-max, max] for each position i
        # Shape: [seq_len, range_size]
        i = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
        relative_pos = i - (i + self.relative_range.unsqueeze(0))  

        # Crop and convert to embedded index.
        relative_pos_clamped = torch.clamp(
            relative_pos,
            -self.max_relative_pos,
            self.max_relative_pos
        )
        relative_pos_ids = relative_pos_clamped + self.max_relative_pos  # [seq_len, range_size]

        # Obtain the embeddings and calculate the average (avoiding the generation of a large seq_len x seq_len matrix).
        relative_emb = self.relative_emb(relative_pos_ids)  # [seq_len, range_size, embed_dim]
        relative_emb_avg = relative_emb.mean(dim=1)  # [seq_len, embed_dim]

        # Broadcast to the batch and add positional encoding.
        x = x + relative_emb_avg.unsqueeze(0)  # [batch_size, seq_len, embed_dim]
        return self.dropout(x)


class DeFiTransformerClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_encoders, hidden_dim, dropout=0.1, local_window_size=10):
        """
        Transformer Classifier: Used for DeFi price manipulation detection
        :param embed_dim: Embedding dimension (dynamic)
        :param num_heads: Number of multi-head attention heads
        :param num_encoders: Number of encoder layers
        :param hidden_dim: Hidden layer dimension of the feedforward network
        :param dropout: Dropout probability
        """
        super().__init__()
        self.num_heads = num_heads
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len=2500, dropout=dropout)
        self.relative_pos_encoding = RelativePositionalEncoding(embed_dim=embed_dim, max_relative_pos=50, dropout=dropout)
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )



    def forward(self, x, attention_mask=None):
        """
        Forward propagation
        :param x: Input embedding (batch_size, seq_len, embed_dim)
        :param attention_mask: Attention mask (batch_size, seq_len)
        :return: logits (batch_size, 1)
        """
        x = self.relative_pos_encoding(x)
        # x = self.pos_encoding(x)

        batch_size, seq_len, embed_dim = x.shape
        mask = None  

        if attention_mask is not None:
            # Constructing the mask matrix: Set the padding positions to -10000 so that the model ignores these positions.
            mask = (1.0 - attention_mask) * -10000.0
            mask = mask.unsqueeze(1).repeat(1, seq_len, 1)
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            mask = mask.view(batch_size * self.num_heads, seq_len, seq_len)

        x = self.transformer_encoder(x, mask=mask)

        # Attention-masked average pooling
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
            seq_len = attention_mask.sum(dim=1, keepdim=True)
            x = x.sum(dim=1) / seq_len  # Average value of valid positions
        else:
            x = x.mean(dim=1)  # Global average pooling
        logits = self.classifier(x)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device, threshold=PRED_THRESHOLD):
    """
    Train the model for one epoch
    :param model: Model instance
    :param loader: Training data loader
    :param criterion: Loss function
    :param optimizer: Optimizer
    :param device: Training device
    :param threshold: Classification threshold
    :return: Average loss, accuracy, F1 score, precision, recall
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    for batch in loader:
        embeddings, labels, attention_mask = batch
        embeddings, labels, attention_mask = (
            embeddings.to(device),
            labels.to(device),
            attention_mask.to(device)
        )
        preds = model(embeddings, attention_mask) 
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents gradient explosion.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Cumulative losses and indicator calculations
        total_loss += loss.item() * embeddings.size(0)
        probs = torch.sigmoid(preds).detach().cpu().numpy()  
        all_probs.extend(probs)
        all_preds.extend((probs >= threshold).astype(int))  
        all_labels.extend(labels.detach().cpu().numpy())
    # Calculate average loss and evaluation metrics.
    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1, precision, recall


def validate(model, loader, criterion, device, threshold=PRED_THRESHOLD):
    """
    Model Validation
    :param model: Model instance
    :param loader: Validation data loader
    :param criterion: Loss function
    :param device: Validation device
    :param threshold: Classification threshold
    :return: Average loss, accuracy, F1 score, precision, recall, AUC, TP, TN, FP, FN
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad(): 
        for batch in loader:
            embeddings, labels, attention_mask = batch
            embeddings, labels, attention_mask = (
                embeddings.to(device),
                labels.to(device),
                attention_mask.to(device)
            )
            preds = model(embeddings, attention_mask)
            loss = criterion(preds, labels)
            total_loss += loss.item() * embeddings.size(0)
            probs = torch.sigmoid(preds).detach().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs >= threshold).astype(int))
            all_labels.extend(labels.detach().cpu().numpy())
    # Calculate evaluation metrics
    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    return avg_loss, acc, f1, precision, recall, auc, tp, tn, fp, fn


def predict_manipulation(model_path, embed_csv_path, device, embed_dim, has_header=False, threshold=PRED_THRESHOLD):
    """
    Single File Prediction: Given an embedding CSV file, predict whether price manipulation exists.
    :param model_path: Path to the saved model
    :param embed_csv_path: Path to the embedding CSV file to be predicted
    :param device: Prediction device
    :param embed_dim: Dynamically obtained embedding dimension
    :param has_header: Whether the CSV file has a header
    :param threshold: Classification threshold
    :return: Manipulation probability, predicted label (1=manipulation exists, 0=no manipulation)
    """

    print(embed_csv_path)
    model = DeFiTransformerClassifier(
        embed_dim=embed_dim,  
        num_heads=NUM_HEADS,
        num_encoders=NUM_ENCODERS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    df = pd.read_csv(embed_csv_path, header=0 if has_header else None)
    embedding_np = df.values

    assert embedding_np.shape[1] == embed_dim, \
        f"The predicted number of columns in the CSV file is incorrect! It should be {embed_dim} columns, but it actually has {embedding_np.shape[1]} columns."
    embedding = torch.tensor(embedding_np, dtype=torch.float32).unsqueeze(0).to(device)
    attention_mask = torch.ones(embedding.shape[:2], device=device)  # Full mask (no padding)

    with torch.no_grad():
        logits = model(embedding, attention_mask)
        prob = torch.sigmoid(logits).item()  
    pred_label = 1 if prob >= threshold else 0
    return prob, pred_label

if __name__ == "__main__":
    negative_sampling = False
    # Loading labels and datasets
    label_dict = load_label_dict(LABEL_PATH)
    full_dataset = DeFiEmbeddingDataset(DATA_DIR, label_dict, EMBED_DIM, has_header=HAS_HEADER)

    # Splitting the data into training and validation sets.
    train_size = int((1 - VAL_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Undersampling of the training set (balancing positive and negative examples)
    train_labels = [full_dataset.label_dict[full_dataset.file_names[i][:-4]] for i in train_indices]
    pos_indices = [i for i, label in zip(train_indices, train_labels) if label == 1]
    neg_indices = [i for i, label in zip(train_indices, train_labels) if label == 0]
    print(f"Original training set - Positive examples: {len(pos_indices)}, Negative examples: {len(neg_indices)}")

    # Sample negative examples proportionally to ensure there are at least some positive examples.
    if len(pos_indices) > 0 and negative_sampling == True:
        target_neg = int(len(pos_indices) / POS_NEG_RATIO)  # Number of target negative examples
        # Ensure that the number of sampled negative examples does not exceed the actual number of negative examples.
        sampled_neg = neg_indices if len(neg_indices) <= target_neg else np.random.choice(neg_indices, target_neg,
                                                                                          replace=False)
        sampled_indices = pos_indices + sampled_neg.tolist()
        train_dataset = Subset(full_dataset, sampled_indices)
        print(f"Training set after sampling - Positive examples: {len(pos_indices)}, Negative examples: {len(sampled_neg)}")

    # Create a data loader.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"Number of samples in the training set: {len(train_dataset)}, number of samples in the validation set: {len(val_dataset)}")
    print(f"Number of CSV files loaded: {len(full_dataset)} (Directory: {DATA_DIR})")

    # Initialize the model and training components.
    model = DeFiTransformerClassifier(
        embed_dim=EMBED_DIM, 
        num_heads=NUM_HEADS,
        num_encoders=NUM_ENCODERS,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(device)

    # Calculate positive sample weights
    train_labels_sampled = [full_dataset.label_dict[full_dataset.file_names[i][:-4]] for i in train_dataset.indices]
    pos_count = sum(train_labels_sampled)
    neg_count = len(train_labels_sampled) - pos_count
    # Handling cases with no positive examples
    if pos_count == 0:
        pos_weight = torch.tensor(1.0, device=device)
    else:
        pos_weight = torch.tensor(neg_count / pos_count, device=device)  
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Weighted loss function
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)  # Adjust weight decay
    # Learning rate scheduler: reduces the learning rate when the validation loss stops improving.
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

    best_val_f1 = 0.0
    early_stop_counter = 0  # Early stop counter

    print("\n" + "=" * 50)
    print(f"Starting training (total {EPOCHS} epochs, classification threshold = {PRED_THRESHOLD})")
    print("=" * 50)
    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_precision, val_recall, val_auc, tp, tn, fp, fn = validate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Training set - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"          Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Validation set - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        print(f"          Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        print(f"          TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")


        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model updated and saved (current best validation F1: {best_val_f1:.4f})")
            early_stop_counter = 0  # Reset the early stopping counter.
        else:
            early_stop_counter += 1
            print(f"Early stop count: {early_stop_counter}/{PATIENCE}")
            if early_stop_counter >= PATIENCE:
                print(f"The early stopping mechanism was triggered, and training stopped at epoch {epoch + 1}.")
                break

    print("\n" + "=" * 50)
    if best_val_f1 != 0.0:
        print(f"Training complete! The optimal model has been saved to...: {MODEL_SAVE_PATH}")
        print(f"Optimal validation F1 score: {best_val_f1:.4f}")
    print("=" * 50)