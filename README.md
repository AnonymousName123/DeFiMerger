# DeFiMerger 🛡️

> An LLM-powered framework for detecting price manipulation attacks in DeFi protocols through multi-modal feature fusion and Transformer-based classification.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

---

## 📖 Overview

**DeFiMerger** is designed to identify malicious price manipulation attacks in decentralized finance (DeFi) protocols. By combining **smart contract source code embeddings** with **transaction event log features**, the system achieves robust detection through advanced fusion techniques and Transformer-based neural networks.

The framework supports **multi-chain analysis** (Ethereum, BSC, Polygon, Arbitrum, Avalanche, Base).

---

## ✨ Core Features

- 🧠 **Multi-Modal Feature Fusion**: Combines contract code embeddings (768D) from LLMs (DeepSeek, Qwen, StarCoder, etc.) with handcrafted event features (256D)
- 🔍 **Advanced Attack Pattern Recognition**: Detects flash loan attacks, wash trading, circular transfers, and oracle manipulation through 80+ engineered features
- 🚀 **State-of-the-Art Architecture**: Transformer encoder with relative positional encoding, attention masking, etc
- 📊 **Cross-Chain Support**: Built-in support for 6 major blockchain networks with automated data collection via Etherscan APIs
- 🎯 **Production-Ready Pipeline**: Complete workflow from raw transaction URLs to trained detection models with 90%+ F1 scores

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch, Transformers (HuggingFace) |
| **ML/Feature Engineering** | scikit-learn, NumPy, Pandas |
| **Code Embeddings** | DeepSeek-Coder, Qwen2.5-Coder, StarCoder, CodeGemma |
| **Data Collection** | BeautifulSoup, Requests, OpenPyXL |
| **Quantization** | BitsAndBytes (4-bit NF4) |

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 12.1+ (for GPU acceleration)

### Installation

```bash
# Clone the repository
git clone https://github.com/AnonymousName123/DeFiMerger.git
cd DeFiMerger

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

```bash
# 1. Prepare dataset.xlsx with columns: Protocol, Chain, Transaction Hash
# 2. Extract transaction URLs
python get_url.py
# 3. Scrape event logs from block explorers
python get_event.py
# 4. Rename protocol folders based on dataset
python change_protocol_name.py
# 5. Parse event data to Excel format
python change_event.py
# 6. Download smart contract source code
# Edit get_code.py to add your Etherscan API keys
python get_code.py
```

### Running the Pipeline

```bash
#### Step 1: Generate Event Embeddings
# Extract 256D features from transaction events
python get_event_embedding.py
#### Step 2: Generate Contract Code Embeddings

# Edit get_contract_embedding.py to specify your model
# Supported models: deepseek-coder-1.3b, Qwen2.5-Coder-1.5B, 
#                   starcoderbase-1b, SmolLM2-1.7B, codegemma-2b

python get_contract_embedding.py

#### Step 3: Fuse Multi-Modal Features
# Combine event + contract embeddings (1024D total)
# Edit merge.py to select fusion method:
# - concatenate (simple)
# - weighted_concat (weighted combination)
# - attention, cross_attention, gated, self_attention_gnn (advanced)
python merge.py

#### Step 4: Train Detection Model
# Train Transformer classifier
python classify.py
# For controlled experiments with fixed test set
python run_RQ2.py
```

---

## 📁 Directory Structure

```
DeFiMerger/
├── dataset/
│   ├── attack incident/          # Confirmed attack cases
│   │   ├── ETH/
│   │   │   └── Protocol_Name/
│   │   │       ├── Event.xlsx    # Parsed event logs
│   │   │       ├── event.txt     # Raw scraped data
│   │   │       ├── source/       # Smart contract .sol files
│   │   │       ├── abi/          # Contract ABIs
│   │   │       └── meta/         # Metadata
│   │   └── BSC/ POL/ ARB/ AVAX/ Base/
│   ├── high_value/               # High-TVL protocols (negative samples)
│   ├──get_url.py                 # Extract URLs from dataset.xlsx
│   ├──get_event.py               # Scrape transaction event logs
│   ├──change_protocol_name.py    # Rename folders by protocol
│   ├──change_event.py            # Parse events to Excel
│   ├──get_code.py                # Download contract source code
│   ├──label.csv                  # Protocol labels (1=attack, 0=safe)
│   └── dataset.xlsx              # Dataset(D1)
│
├── event_feature/
│   ├── Feature Engineering/      # Confirmed attack cases
│   │   └── get_event_embedding.py# Generate event features (256D)
│   ├── Event Classification/     # Confirmed attack cases
│   │   ├── event_categories.json # Event type taxonomy
│   │   └── get_event_types.py    # Get Event Types
│   └── embeddings/               # 256D event embeddings
│       ├── attack incident/
│       └── high_value/
├── 
│
├── contract_feature/
│   ├──get_contract_embedding.py  # Generate code embeddings (768D)
│   └── embeddings/
│       └── Starcoder_768/        # 768D contract embeddings
│           ├── attack incident/
│           └── high_value/
│
├── merged_features/
│   └── concatenate/              # Fused 1024D features
│       └── Starcoder_768/
│
├── merge.py                      # Fuse multi-modal features
├── classify.py                   # Train/test Transformer model
├── run_RQ2.py                    # RQ2 experiment
└── defi_manipulation_model.pth   # Trained model weights
```

---

## ⚙️ Configuration

### API Keys (get_code.py)
```python
# Add your Etherscan API keys for each chain
API_KEYS = [
    "YOUR_ETHERSCAN_API_KEY_1",
    "YOUR_ETHERSCAN_API_KEY_2",
    # Rotate multiple keys to avoid rate limits
]
```

### Model Selection (get_contract_embedding.py)
```python
# Choose your embedding model
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
# Alternatives:
# - "Qwen/Qwen2.5-Coder-1.5B"
# - "bigcode/starcoderbase-1b"
# - "google/codegemma-2b"
# - "HuggingFaceTB/SmolLM2-1.7B-Instruct"
```

### Training Hyperparameters (classify.py)
```python
BATCH_SIZE = 4          # Adjust based on GPU memory
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_ENCODERS = 2        # Transformer layers
NUM_HEADS = 8           # Attention heads
DROPOUT = 0.3
PRED_THRESHOLD = 0.1    # Classification threshold
```

### Fusion Method (merge.py)
```python
fusion_methods = [
    'concatenate',           # Simple stacking
    'weighted_concat',       # Weighted combination (event=0.3, code=0.7)
    'attention',             # Multi-head attention fusion
    'cross_attention',       # Cross-modal attention
    'gated',                 # Gating mechanism
    'self_attention_gnn'     # Graph + attention fusion
]
```

---


## 🔬 Model Architecture

```
Input: [Event Features (256D)] + [Contract Embeddings (768D)]
         ↓
   Fusion Layer (1024D)
         ↓
   Relative Positional Encoding
         ↓
   Transformer Encoder (2 layers, 8 heads)
         ↓
   Transformer Dncoder
         ↓
   Sigmoid → Binary Prediction
```

**Key Features:**
- **Variable-length sequences** handled via dynamic padding + attention masks
- **Relative positional encoding** for capturing sequential dependencies
- **Class imbalance mitigation** through BCEWithLogitsLoss + pos_weight
- **Early stopping** with patience=3 to prevent overfitting

---

## 📊 Performance Metrics

| RQs                         | Accuracy     | Precision    | Recall      | F1-Score     | AUC          |
|-----------------------------|--------------|--------------|-------------|--------------|--------------|
| RQ1 (Detection performance) | 99.34 ± 0.52 | 96.01 ± 3.14 | 100 ± 0.00  | 97.94 ± 1.65 | 99.86 ± 0.10 |

*Each experiment is repeated for ten independent runs. We report the average results along with
their standard deviations to account for potential variance. *

---

## 🙏 Acknowledgments

- **Etherscan API** for blockchain data access
- **HuggingFace** for pre-trained code models
- **PyTorch Team** for deep learning framework

---



**⭐ If this project helps your research or business, please consider giving it a star!**
