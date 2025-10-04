# Water Potability Classification: Deep Learning Models Comparison

## Table of Contents
1. [Overview](#overview)
2. [Code Structure](#code-structure)
3. [Dataset Requirements](#dataset-requirements)
4. [Installation](#installation)
5. [Configuration Parameters](#configuration-parameters)
6. [Implementation Details](#implementation-details)
7. [Model Architectures](#model-architectures)
8. [Training Pipeline](#training-pipeline)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Explainability with SHAP](#explainability-with-shap)
11. [Output Artifacts](#output-artifacts)
12. [Usage Guide](#usage-guide)
13. [Code Walkthrough](#code-walkthrough)
14. [Troubleshooting](#troubleshooting)

---

## Overview

This Jupyter notebook implements a comprehensive machine learning pipeline for binary classification of water potability. The code compares three state-of-the-art architectures:

- **Multi-Layer Perceptron (MLP)** - Traditional deep learning baseline
- **TabNet** - Attention-based architecture designed for tabular data
- **Feature Tokenizer Transformer (FT-Transformer)** - Transformer architecture adapted for numerical features

### Key Features

- **5-fold Stratified Cross-Validation** for robust performance estimation
- **SHAP Explainability** integrated for all three models
- **Comprehensive Visualization Suite** including EDA, training curves, confusion matrices, ROC curves
- **Automatic Model Selection** based on ROC-AUC score
- **Full Data Retraining** of the best model for deployment
- **Artifact Persistence** (scaler, trained models)
- **GPU Acceleration** with automatic CPU fallback

---

## Code Structure

```
water-potability-classification/
│
├── Untitled23 (2).ipynb          # Main implementation notebook
├── water_quality_potability.csv  # Dataset (required)
├── artifacts/                    # Generated after running
│   ├── scaler.pkl               # Fitted StandardScaler
│   ├── mlp_full.pth            # Best MLP model (if selected)
│   ├── ft_transformer_full.pth # Best FT-Transformer (if selected)
│   └── tabnet_full.zip         # Best TabNet model (if selected)
└── README.md                    # This file
```

---

## Dataset Requirements

### Expected Format
- **File Name**: `water_quality_potability.csv`
- **Format**: CSV with header row
- **Target Column**: `Potability` (binary: 0 or 1)
- **Feature Columns**: Numerical features (pH, hardness, solids, etc.)

### Example Schema
```
ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability
7.0, 204.0, 20791.0, 7.3, 368.5, 564.3, 10.4, 86.9, 2.9, 0
```

### Data Preprocessing Applied
1. **Feature Extraction**: All columns except `Potability` are treated as features
2. **Standardization**: StandardScaler (mean=0, std=1) applied to all features
3. **Missing Values**: The code assumes clean data (add imputation if needed)

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA-enabled GPU (optional but recommended)
- 8GB+ RAM

### Required Packages

```bash
# Core deep learning
pip install torch torchvision torchaudio

# Tabular models
pip install pytorch-tabnet

# Explainability
pip install shap

# Data science stack
pip install scikit-learn pandas numpy

# Visualization
pip install matplotlib seaborn

# Model persistence
pip install joblib
```

### Quick Install (all packages)
```bash
pip install pytorch-tabnet shap torch torchvision torchaudio scikit-learn matplotlib seaborn pandas numpy joblib
```

### Google Colab Setup
The notebook includes Colab-specific installation cells:
```python
!pip -q install pytorch-tabnet shap torch torchvision torchaudio --upgrade
!pip -q install scikit-learn matplotlib seaborn pandas numpy joblib
```

---

## Configuration Parameters

All hyperparameters are defined in the **Settings** section:

```python
# Reproducibility
RANDOM_SEED = 42                # Fixed seed for reproducibility

# Data
DATA_PATH = "water_quality_potability.csv"  # Dataset location

# Training
BATCH_SIZE = 128                # Batch size for all models
LR = 1e-3                       # Learning rate (0.001)

# Model-specific epochs
MLP_EPOCHS = 40                 # MLP training epochs
FTT_EPOCHS = 50                 # FT-Transformer epochs
TABNET_MAX_EPOCHS = 100         # TabNet max epochs (early stopping enabled)

# Cross-validation
N_SPLITS = 5                    # Number of CV folds

# Output
ARTIFACT_DIR = "artifacts"      # Directory for saved models/scaler

# Device
DEVICE = "cuda" or "cpu"        # Automatically detected
```

### Modifying Parameters

To adjust hyperparameters, change values in the Settings section:

```python
# Example: Train longer with smaller batches
BATCH_SIZE = 64
MLP_EPOCHS = 100
FTT_EPOCHS = 150
```

---

## Implementation Details

### 1. Imports and Dependencies

```python
# Core libraries
import numpy as np              # Numerical operations
import pandas as pd             # Data manipulation
import torch                    # Deep learning framework
import torch.nn as nn           # Neural network modules
import torch.nn.functional as F # Activation functions

# Scikit-learn
from sklearn.model_selection import StratifiedKFold     # CV splitting
from sklearn.preprocessing import StandardScaler        # Feature scaling
from sklearn.metrics import (                           # Evaluation metrics
    accuracy_score, f1_score, roc_auc_score,
    roc_curve, classification_report, confusion_matrix
)

# Specialized libraries
from pytorch_tabnet.tab_model import TabNetClassifier  # TabNet implementation
import shap                                             # Model explainability

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Reproducibility Setup

The code ensures reproducibility through fixed random seeds:

```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)          # NumPy operations
random.seed(RANDOM_SEED)             # Python random module
torch.manual_seed(RANDOM_SEED)       # PyTorch CPU operations
# Note: torch.cuda.manual_seed_all(RANDOM_SEED) can be added for multi-GPU
```

### 3. Device Configuration

Automatic GPU detection with CPU fallback:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
```

### 4. Data Loading

```python
df = pd.read_csv(DATA_PATH)
assert "Potability" in df.columns, "Dataset must have 'Potability' column."

features = df.columns.tolist()
features.remove("Potability")        # Extract feature names
X_raw_all = df[features].copy()      # Feature matrix
y_all = df["Potability"].values      # Target labels
```

---

## Model Architectures

### 1. Multi-Layer Perceptron (MLP)

**Architecture Details:**
```python
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)    # Input → 128 neurons
        self.bn1 = nn.BatchNorm1d(128)          # Batch normalization
        self.fc2 = nn.Linear(128, 64)           # 128 → 64 neurons
        self.bn2 = nn.BatchNorm1d(64)           # Batch normalization
        self.drop = nn.Dropout(0.3)             # 30% dropout
        self.fc3 = nn.Linear(64, 32)            # 64 → 32 neurons
        self.out = nn.Linear(32, 2)             # 32 → 2 classes
```

**Forward Pass:**
```python
def forward(self, x):
    x = F.relu(self.bn1(self.fc1(x)))          # Layer 1 + BN + ReLU
    x = self.drop(F.relu(self.bn2(self.fc2(x))))  # Layer 2 + BN + ReLU + Dropout
    x = F.relu(self.fc3(x))                    # Layer 3 + ReLU
    return self.out(x)                         # Output logits
```

**Total Parameters:** ~19,000 (varies with input dimension)

**Key Design Choices:**
- Batch Normalization for stable training
- Dropout for regularization
- ReLU activation for non-linearity
- Gradual dimension reduction (128 → 64 → 32)

---

### 2. TabNet

**Implementation:**
```python
from pytorch_tabnet.tab_model import TabNetClassifier

tabnet = TabNetClassifier(
    seed=RANDOM_SEED,
    verbose=0
)

tabnet.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=TABNET_MAX_EPOCHS,     # 100
    patience=15,                       # Early stopping
    batch_size=BATCH_SIZE,            # 128
    virtual_batch_size=32             # For ghost batch normalization
)
```

**Architecture Components:**
- **Feature Transformer**: Processes features through decision steps
- **Attentive Transformer**: Learns which features to use at each step
- **Decision Steps**: Sequential attention mechanism (default: 3 steps)
- **Ghost Batch Normalization**: Stabilizes training

**Key Advantages:**
- Built-in feature importance
- Interpretable attention masks
- Sparse feature selection
- Designed specifically for tabular data

---

### 3. Feature Tokenizer Transformer (FT-Transformer)

**Custom Implementation:**

#### FeatureTokenizer Module
```python
class FeatureTokenizer(nn.Module):
    def __init__(self, n_num_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num_features, d_token) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_num_features, d_token))
        self.cls = nn.Parameter(torch.randn(1, d_token) * 0.02)  # CLS token

    def forward(self, x):
        # x: (batch, n_features) → (batch, n_features, d_token)
        x = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        # Prepend CLS token
        cls = self.cls.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1)  # (batch, 1+n_features, d_token)
```

#### FTTransformer Module
```python
class FTTransformer(nn.Module):
    def __init__(self, n_num_features, d_token=64, n_heads=8,
                 n_layers=3, dropout=0.1, n_classes=2):
        super().__init__()
        self.tok = FeatureTokenizer(n_num_features, d_token)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token,              # 64
            nhead=n_heads,                # 8 attention heads
            dim_feedforward=d_token*4,    # 256
            dropout=dropout,              # 0.1
            batch_first=True,
            activation="gelu",            # GELU activation
            norm_first=True               # Pre-layer normalization
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Classification head
        self.norm = nn.LayerNorm(d_token)
        self.head = nn.Sequential(
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, n_classes)
        )

    def forward(self, x):
        tok = self.tok(x)                  # Tokenize features
        enc = self.encoder(tok)            # Apply transformer
        cls = self.norm(enc[:, 0, :])      # Extract CLS token
        return self.head(cls)              # Classification
```

**Architecture Specifications:**
- **Token Dimension**: 64
- **Attention Heads**: 8
- **Encoder Layers**: 3
- **Feedforward Dimension**: 256 (4 × token_dim)
- **Dropout**: 0.1
- **Activation**: GELU (Gaussian Error Linear Unit)

**Total Parameters:** ~50,000-100,000 (varies with input features)

---

## Training Pipeline

### 1. PyTorch Training Function

```python
def train_pytorch(model, train_loader, epochs=20, lr=1e-3, device=DEVICE, name="Model"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []

    for ep in range(epochs):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)

        epoch_loss = running / len(train_loader.dataset)
        losses.append(epoch_loss)

        if (ep+1) % max(1, (epochs//4)) == 0:
            print(f"{name} Epoch {(ep+1)}/{epochs} loss: {epoch_loss:.5f}")

    return model, losses
```

**Training Configuration:**
- **Optimizer**: Adam (adaptive learning rate)
- **Loss Function**: CrossEntropyLoss (for binary classification)
- **Learning Rate**: 0.001 (default)
- **Batch Processing**: Mini-batch gradient descent

### 2. Cross-Validation Loop

```python
scaler_global = StandardScaler()
X_all = scaler_global.fit_transform(X_raw_all.values)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
    print(f"\n========== Fold {fold_idx}/{N_SPLITS} ==========")
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # Create PyTorch datasets
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Train each model...
```

**Cross-Validation Strategy:**
- **Stratified Splitting**: Preserves class distribution in each fold
- **5 Folds**: 80% training, 20% testing per fold
- **Shuffling**: Enabled with fixed random state
- **Per-Fold Training**: Each model trained independently per fold

### 3. Per-Fold Training Workflow

For each fold, the following steps are executed:

#### MLP Training
```python
mlp = MLP(X_train.shape[1])
mlp, mlp_losses = train_pytorch(mlp, train_loader, epochs=MLP_EPOCHS,
                                 lr=LR, device=DEVICE, name=f"MLP (fold {fold_idx})")

# Training loss visualization
plt.figure()
plt.plot(mlp_losses)
plt.title(f"MLP Training Loss (fold {fold_idx})")
plt.xlabel("Epoch")
plt.show()
```

#### TabNet Training
```python
tabnet = TabNetClassifier(seed=RANDOM_SEED, verbose=0)
start = time.time()
tabnet.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    max_epochs=TABNET_MAX_EPOCHS,
    patience=15,                  # Early stopping
    batch_size=BATCH_SIZE,
    virtual_batch_size=32
)
print("TabNet runtime (s):", time.time()-start)
```

#### FT-Transformer Training
```python
ftt = FTTransformer(
    n_num_features=X_train.shape[1],
    d_token=64,
    n_heads=8,
    n_layers=3,
    dropout=0.1,
    n_classes=2
).to(DEVICE)

ftt, ftt_losses = train_pytorch(ftt, train_loader, epochs=FTT_EPOCHS,
                                 lr=LR, device=DEVICE, name=f"FT-Transformer (fold {fold_idx})")
```

---

## Evaluation Metrics

### 1. Evaluation Function

```python
def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model", plot_roc=False, ax=None):
    # Classification report
    print(f"\n-- {model_name} Evaluation --")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrices
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, title=f"{model_name} Confusion Matrix")
    plot_confusion(cm, title=f"{model_name} Confusion Matrix (Normalized)", normalize=True)

    # ROC-AUC
    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            if plot_roc:
                # ROC curve plotting logic...
        except Exception:
            auc = None

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {"accuracy": acc, "f1": f1, "auc": auc}
```

### 2. Metrics Computed

#### Per-Fold Metrics
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

#### Aggregate Metrics
```python
def aggregate_results(res_list):
    df = pd.DataFrame(res_list)
    return df.mean(numeric_only=True).to_dict(), df

# Compute mean metrics across folds
for model_name, res_list in results_per_model.items():
    mean_metrics, df_detail = aggregate_results(res_list)
    agg_summary[model_name] = mean_metrics
    detailed_frames[model_name] = df_detail

summary_df = pd.DataFrame(agg_summary).T
display(summary_df)
```

### 3. Confusion Matrix Visualization

```python
def plot_confusion(cm, title="Confusion matrix", normalize=False):
    plt.figure(figsize=(4.2,3.6))
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm = cm / (cm_sum + 1e-9)
        fmt = ".2f"
    else:
        fmt = "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues")
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
```

### 4. ROC Curve Aggregation

```python
plt.figure(figsize=(6,5))
for model_name, curves in roc_curves.items():
    aucs = []
    for fpr, tpr, auc in curves:
        aucs.append(auc if auc is not None else np.nan)
        plt.plot(fpr, tpr, alpha=0.3, label=None)

    label = f"{model_name} (mean AUC={np.nanmean(aucs):.4f})"
    plt.plot([], [], label=label)

plt.plot([0,1],[0,1], 'k--', label="Random")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curves across folds")
plt.legend()
plt.show()
```

---

## Explainability with SHAP

### 1. SHAP for MLP and FT-Transformer

Uses **GradientExplainer** for gradient-based models:

```python
# Select background and test samples
be_n = min(100, X_train.shape[0])    # Background samples
te_n = min(100, X_test.shape[0])     # Test samples

background = torch.tensor(
    X_train[np.random.choice(X_train.shape[0], be_n, replace=False)],
    dtype=torch.float32
).to(DEVICE)

test_sample = torch.tensor(X_test[:te_n], dtype=torch.float32).to(DEVICE)

# Create explainer
explainer_mlp = shap.GradientExplainer(mlp, background)
shap_vals = explainer_mlp.shap_values(test_sample)

# Visualize
shap.summary_plot(shap_vals[1], X_test[:te_n], feature_names=features, show=True)
```

**Why Subset-Based?**
- Full SHAP computation is computationally expensive
- 100-200 samples provide good approximation
- Enables GPU acceleration

### 2. SHAP for TabNet

Uses **model-agnostic Explainer**:

```python
be_n = min(200, X_train.shape[0])
te_n = min(100, X_test.shape[0])

explainer_tab = shap.Explainer(
    tabnet.predict_proba,
    X_train[np.random.choice(X_train.shape[0], be_n, replace=False)]
)

shap_tab = explainer_tab(X_test[:te_n])
shap.summary_plot(shap_tab, X_test[:te_n], feature_names=features, show=True)
```

**Fallback Mechanism:**
If SHAP fails, uses TabNet's built-in feature importances:

```python
try:
    imp = tabnet.feature_importances_
    imp_df = pd.DataFrame({
        "feature": features,
        "importance": imp
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(6,3))
    sns.barplot(x="importance", y="feature", data=imp_df)
    plt.title("TabNet feature importances")
    plt.show()
except:
    pass
```

### 3. SHAP Summary Plot Interpretation

The summary plot shows:
- **Y-axis**: Features ranked by importance
- **X-axis**: SHAP value (impact on model output)
- **Color**: Feature value (red = high, blue = low)
- **Width**: Density of data points

**Reading the Plot:**
- Features at top are most important
- Positive SHAP values increase prediction probability
- Negative SHAP values decrease prediction probability
- Color gradient shows feature value correlation with impact

---

## Output Artifacts

### 1. Saved Files

After training, the following artifacts are saved in `artifacts/`:

#### Scaler
```python
import joblib
joblib.dump(scaler_global, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
```

**Usage:**
```python
scaler = joblib.load("artifacts/scaler.pkl")
X_new_scaled = scaler.transform(X_new)
```

#### MLP Model
```python
torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "mlp_full.pth"))
```

**Usage:**
```python
model = MLP(input_dim=9)
model.load_state_dict(torch.load("artifacts/mlp_full.pth"))
model.eval()
```

#### FT-Transformer Model
```python
torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "ft_transformer_full.pth"))
```

**Usage:**
```python
model = FTTransformer(n_num_features=9, d_token=64, n_heads=8, n_layers=3)
model.load_state_dict(torch.load("artifacts/ft_transformer_full.pth"))
model.eval()
```

#### TabNet Model
```python
model.save_model(os.path.join(ARTIFACT_DIR, "tabnet_full.zip"))
```

**Usage:**
```python
from pytorch_tabnet.tab_model import TabNetClassifier
model = TabNetClassifier()
model.load_model("artifacts/tabnet_full.zip")
```

### 2. Model Selection Logic

```python
best_model_name = None
if summary_df["auc"].notnull().any():
    best_model_name = summary_df["auc"].idxmax()  # Select by AUC
else:
    best_model_name = summary_df["f1"].idxmax()   # Fallback to F1

print(f"\n✅ Best model by CV primary metric: {best_model_name}")
```

**Selection Priority:**
1. ROC-AUC (primary metric)
2. F1-Score (fallback if AUC unavailable)

### 3. Full Dataset Retraining

The best model is retrained on the entire dataset:

```python
if best_model_name == "MLP":
    full_ds = TensorDataset(
        torch.tensor(X_all, dtype=torch.float32),
        torch.tensor(y_all, dtype=torch.long)
    )
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True)
    model = MLP(X_all.shape[1]).to(DEVICE)
    model, _ = train_pytorch(model, full_loader, epochs=MLP_EPOCHS,
                             lr=LR, device=DEVICE, name="MLP (full)")
    torch.save(model.state_dict(), os.path.join(ARTIFACT_DIR, "mlp_full.pth"))
```

---

## Usage Guide

### 1. Running the Notebook

#### Step 1: Prepare Dataset
Place `water_quality_potability.csv` in the notebook directory.

#### Step 2: Install Dependencies
```python
!pip -q install pytorch-tabnet shap torch torchvision torchaudio --upgrade
!pip -q install scikit-learn matplotlib seaborn pandas numpy joblib
```

#### Step 3: Run All Cells
Execute all cells sequentially. Total runtime:
- **CPU**: 30-60 minutes
- **GPU**: 10-20 minutes

#### Step 4: Review Outputs
- EDA visualizations
- Training loss curves (15 plots: 5 folds × 3 models)
- Confusion matrices (30 plots: 5 folds × 3 models × 2 types)
- SHAP summary plots (up to 15 plots)
- Aggregate ROC curve
- Cross-validation summary table

#### Step 5: Check Artifacts
```bash
ls artifacts/
# Output: scaler.pkl, mlp_full.pth (or tabnet_full.zip, ft_transformer_full.pth)
```

### 2. Using Trained Models for Prediction

#### Complete Inference Pipeline

```python
import joblib
import torch
import numpy as np
import pandas as pd

# Load scaler
scaler = joblib.load("artifacts/scaler.pkl")

# Load model (example: MLP)
model = MLP(input_dim=9)  # Adjust based on your feature count
model.load_state_dict(torch.load("artifacts/mlp_full.pth"))
model.eval()

# Prepare new data
new_data = pd.DataFrame({
    'ph': [7.0],
    'Hardness': [204.0],
    'Solids': [20791.0],
    'Chloramines': [7.3],
    'Sulfate': [368.5],
    'Conductivity': [564.3],
    'Organic_carbon': [10.4],
    'Trihalomethanes': [86.9],
    'Turbidity': [2.9]
})

# Preprocess
X_new_scaled = scaler.transform(new_data.values)
X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

# Predict
with torch.no_grad():
    output = model(X_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = output.argmax(dim=1).item()
    confidence = probabilities[0, prediction].item()

print(f"Prediction: {'Potable' if prediction == 1 else 'Not Potable'}")
print(f"Confidence: {confidence:.2%}")
```

### 3. Batch Prediction

```python
# Load multiple samples
new_data = pd.read_csv("new_water_samples.csv")
X_new_scaled = scaler.transform(new_data.values)
X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)

with torch.no_grad():
    output = model(X_tensor)
    predictions = output.argmax(dim=1).numpy()
    probabilities = torch.softmax(output, dim=1).numpy()

results_df = pd.DataFrame({
    'prediction': predictions,
    'prob_not_potable': probabilities[:, 0],
    'prob_potable': probabilities[:, 1]
})

results_df.to_csv("predictions.csv", index=False)
```

---

## Code Walkthrough

### Section 1: Setup and Configuration (Lines 1-35)

**Purpose:** Install dependencies, import libraries, set seeds, configure device

**Key Operations:**
- Install PyTorch, TabNet, SHAP
- Import 15+ libraries
- Set `RANDOM_SEED = 42` for reproducibility
- Detect CUDA availability
- Create `artifacts/` directory

**Customization Points:**
- Change `DATA_PATH` if dataset has different name
- Adjust `BATCH_SIZE`, `LR`, `N_SPLITS`
- Modify epoch counts

---

### Section 2: Data Loading (Lines 36-50)

**Purpose:** Load CSV and prepare features/labels

**Operations:**
```python
df = pd.read_csv(DATA_PATH)              # Load CSV
features = df.columns.tolist()           # Extract column names
features.remove("Potability")            # Remove target
X_raw_all = df[features].copy()          # Feature matrix
y_all = df["Potability"].values          # Target vector
```

**Validation:**
- Asserts `Potability` column exists
- Displays first 5 rows
- Prints dataset shape

---

### Section 3: Exploratory Data Analysis (Lines 51-95)

**Purpose:** Visualize data distributions and relationships

**Analyses Performed:**

1. **Target Distribution**
```python
print(df["Potability"].value_counts())
sns.countplot(x="Potability", data=df)
```

2. **Correlation Heatmap**
```python
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
```

3. **Top Features by Variance**
```python
top_feats = df[features].var().sort_values(ascending=False).index[:6]
```

4. **KDE Plots**
```python
for col in top_feats:
    sns.kdeplot(data=df, x=col, hue="Potability", fill=True)
```

5. **Pairplot**
```python
sns.pairplot(df[top_4_features + ["Potability"]], hue="Potability")
```

---

### Section 4: Utility Functions (Lines 96-150)

**Two main utilities:**

#### plot_confusion()
- Plots confusion matrix as heatmap
- Supports normalization
- Customizable title

#### evaluate_model()
- Prints classification report
- Plots confusion matrices (normal + normalized)
- Computes ROC-AUC
- Returns metrics dictionary

---

### Section 5: Model Definitions (Lines 151-250)

**Three classes defined:**

1. **MLP** (Lines 151-165)
   - 4-layer feedforward network
   - BatchNorm + Dropout regularization

2. **train_pytorch()** (Lines 166-190)
   - Generic PyTorch training loop
   - Adam optimizer + CrossEntropyLoss
   - Returns model and loss history

3. **FeatureTokenizer** (Lines 191-210)
   - Converts numerical features to tokens
   - Adds CLS token

4. **FTTransformer** (Lines 211-250)
   - Transformer encoder stack
   - Classification head

---

### Section 6: Cross-Validation Loop (Lines 251-450)

**Main training pipeline:**

```python
for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all), 1):
    # 1. Split data
    X_train, X_test = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]

    # 2. Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    # 3. Train MLP
    mlp = MLP(X_train.shape[1])
    mlp, mlp_losses = train_pytorch(mlp, train_loader, epochs=40, ...)
    # Evaluate + SHAP

    # 4. Train TabNet
    tabnet = TabNetClassifier(...)
    tabnet.fit(X_train, y_train, ...)
    # Evaluate + SHAP

    # 5. Train FT-Transformer
    ftt = FTTransformer(...)
    ftt, ftt_losses = train_pytorch(ftt, train_loader, epochs=50, ...)
    # Evaluate + SHAP
```

**Per-Fold Per-Model:**
- Training
- Evaluation (classification report, confusion matrix)
- SHAP explanation
- Results storage

---

### Section 7: Results Aggregation (Lines 451-480)

**Operations:**

1. **Aggregate metrics across folds**
```python
def aggregate_results(res_list):
    df = pd.DataFrame(res_list)
    return df.mean(numeric_only=True).to_dict(), df
```

2. **Create summary DataFrame**
```python
summary_df = pd.DataFrame(agg_summary).T
display(summary_df)
```

3. **Plot combined ROC curves**
```python
for model_name, curves in roc_curves.items():
    # Plot all fold curves
    # Annotate with mean AUC
```

---

### Section 8: Model Selection and Saving (Lines 481-530)

**Logic:**

1. **Select best model**
```python
if summary_df["auc"].notnull().any():
    best_model_name = summary_df["auc"].idxmax()
else:
    best_model_name = summary_df["f1"].idxmax()
```

2. **Save scaler**
```python
joblib.dump(scaler_global, "artifacts/scaler.pkl")
```

3. **Retrain best model on full data**
```python
if best_model_name == "MLP":
    # Create full dataset loader
    # Train MLP on entire dataset
    # Save weights
elif best_model_name == "FT-Transformer":
    # Train FT-Transformer on entire dataset
    # Save weights
else:  # TabNet
    # Train TabNet on entire dataset
    # Save model
```

---

## Exploratory Data Analysis

### 1. Class Distribution Analysis

**Code:**
```python
print("\nTarget distribution:")
print(df["Potability"].value_counts())

plt.figure(figsize=(5,3))
sns.countplot(x="Potability", data=df)
plt.title("Class distribution (Potability)")
plt.show()
```

**What to Look For:**
- Class imbalance (if 70/30 or worse, consider SMOTE/class weights)
- Total sample count

---

### 2. Feature Correlation Analysis

**Code:**
```python
plt.figure(figsize=(9,7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation heatmap")
plt.show()
```

**What to Look For:**
- High correlations (> 0.8) suggest multicollinearity
- Negative correlations with target indicate inverse relationships
- Clusters of correlated features

---

### 3. Feature Distribution Analysis

**Code:**
```python
top_feats = df[features].var().sort_values(ascending=False).index[:6].tolist()

for col in top_feats:
    plt.figure(figsize=(6,3.2))
    sns.kdeplot(data=df, x=col, hue="Potability", fill=True, common_norm=False)
    plt.title(f"Distribution of {col} by Potability")
    plt.tight_layout()
    plt.show()
```

**What to Look For:**
- Clear separation between classes indicates discriminative feature
- Overlapping distributions suggest less predictive power
- Outliers or skewed distributions

---

### 4. Multivariate Analysis

**Code:**
```python
pp_feats = top_feats[:4]
sns.pairplot(df[pp_feats + ["Potability"]], hue="Potability", diag_kind="kde", corner=True)
plt.suptitle("Pairplot (top 4 features)", y=1.02)
plt.show()
```

**What to Look For:**
- Linear relationships between features
- Class separability in 2D feature space
- Diagonal KDE plots show individual distributions

---

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size
BATCH_SIZE = 64  # or 32

# Use CPU instead
DEVICE = torch.device("cpu")

# Clear cache between folds
torch.cuda.empty_cache()
```

---

#### 2. Dataset Not Found

**Error:**
```
FileNotFoundError: water_quality_potability.csv
```

**Solution:**
```python
# Check file location
import os
print(os.listdir())

# Update path
DATA_PATH = "/path/to/water_quality_potability.csv"

# Or in Colab
from google.colab import files
uploaded = files.upload()
DATA_PATH = list(uploaded.keys())[0]
```

---

#### 3. SHAP Computation Takes Too Long

**Issue:** SHAP hangs or takes > 10 minutes

**Solution:**
```python
# Reduce subset sizes
be_n = min(50, X_train.shape[0])   # Background: 100 → 50
te_n = min(50, X_test.shape[0])    # Test: 100 → 50

# Or skip SHAP for faster runs
# Comment out SHAP sections in the CV loop
```

---

#### 4. Low Model Performance

**Issue:** All models achieve < 60% accuracy

**Solutions:**

1. **Check data quality**
```python
# Missing values
print(df.isnull().sum())

# Fill missing values
df.fillna(df.median(), inplace=True)
```

2. **Class imbalance**
```python
from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Use in loss function
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
```

3. **Feature engineering**
```python
# Create interaction features
df['ph_hardness'] = df['ph'] * df['Hardness']
df['solids_conductivity'] = df['Solids'] * df['Conductivity']
```

---

#### 5. Model Doesn't Save

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'artifacts/mlp_full.pth'
```

**Solution:**
```python
# Create directory with proper permissions
import os
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Use absolute path
import os.path
full_path = os.path.abspath(ARTIFACT_DIR)
torch.save(model.state_dict(), os.path.join(full_path, "mlp_full.pth"))
```

---

#### 6. TabNet Import Error

**Error:**
```
ModuleNotFoundError: No module named 'pytorch_tabnet'
```

**Solution:**
```bash
pip install pytorch-tabnet

# If still fails, upgrade PyTorch first
pip install torch --upgrade
pip install pytorch-tabnet
```

---

## Performance Optimization Tips

### 1. Speed Up Training

```python
# Use DataLoader num_workers
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                         shuffle=True, num_workers=4, pin_memory=True)

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Use mixed precision (GPU only)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(xb)
    loss = loss_fn(output, yb)
```

### 2. Reduce Memory Usage

```python
# Gradient accumulation
accumulation_steps = 4
for i, (xb, yb) in enumerate(train_loader):
    output = model(xb)
    loss = loss_fn(output, yb) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        opt.step()
        opt.zero_grad()

# Delete unused variables
del mlp, mlp_losses
torch.cuda.empty_cache()
```

### 3. Parallel Fold Execution

```python
# Not implemented in current code, but can be added with joblib
from joblib import Parallel, delayed

def train_fold(fold_idx, train_idx, test_idx):
    # Fold training logic
    return results

results = Parallel(n_jobs=N_SPLITS)(
    delayed(train_fold)(i, train_idx, test_idx)
    for i, (train_idx, test_idx) in enumerate(skf.split(X_all, y_all))
)
```

---

## Advanced Customization

### 1. Adding New Models

```python
# Define custom model
class CustomModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Architecture definition

    def forward(self, x):
        # Forward pass
        return output

# Add to CV loop
custom = CustomModel(X_train.shape[1])
custom, custom_losses = train_pytorch(custom, train_loader, epochs=40, ...)
# Evaluate
custom_res = evaluate_model(y_test, preds, probs, model_name="Custom")
results_per_model["Custom"].append(custom_res)
```

### 2. Hyperparameter Tuning

```python
# Grid search example (manual)
param_grid = {
    'lr': [1e-4, 1e-3, 1e-2],
    'batch_size': [64, 128, 256],
    'dropout': [0.1, 0.3, 0.5]
}

best_score = 0
best_params = {}

for lr in param_grid['lr']:
    for bs in param_grid['batch_size']:
        for dropout in param_grid['dropout']:
            # Train with these params
            # Evaluate
            if score > best_score:
                best_score = score
                best_params = {'lr': lr, 'batch_size': bs, 'dropout': dropout}
```

### 3. Custom Loss Functions

```python
# Focal loss for imbalanced data
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

# Use in training
loss_fn = FocalLoss(alpha=1, gamma=2)
```

---

## Code Metrics

### Lines of Code
- **Total**: ~530 lines
- **Imports**: ~35 lines
- **Configuration**: ~15 lines
- **EDA**: ~45 lines
- **Utilities**: ~55 lines
- **Models**: ~100 lines
- **Training Loop**: ~280 lines
- **Aggregation/Saving**: ~50 lines

### Time Complexity
- **Data Loading**: O(n)
- **Standardization**: O(n × m) where m = features
- **CV Splitting**: O(n × log n)
- **Per-Fold Training**: O(epochs × n/batch_size × model_complexity)
- **SHAP**: O(background_samples × test_samples × features)

### Space Complexity
- **Dataset**: O(n × m)
- **Model Parameters**: O(model_size)
- **Gradients**: O(model_size)
- **Activations**: O(batch_size × hidden_dims)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{water_potability_ml,
  author = {Your Name},
  title = {Water Potability Classification using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/water-potability}}
}
```

### Referenced Papers

1. **TabNet**: Arik, S. Ö., & Pfister, T. (2021). TabNet: Attentive Interpretable Tabular Learning. AAAI.

2. **FT-Transformer**: Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS.

3. **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.

---

## License

This code is provided for educational and research purposes.

---

## Contact

For questions or issues, please open an issue on GitHub or contact the author.

---

**Last Updated:** October 2025
**Version:** 1.0
**Tested On:** Python 3.8+, PyTorch 2.0+, CUDA 11.8
