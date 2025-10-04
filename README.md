# Water Potability Prediction: A Comparative Analysis of Deep Learning and Tabular Machine Learning Models

## Abstract

This repository contains the implementation and experimental framework for a comprehensive study on water potability classification using advanced deep learning and tabular machine learning techniques. The research compares three state-of-the-art models: Multi-Layer Perceptron (MLP), TabNet, and Feature Tokenizer Transformer (FT-Transformer), with integrated SHAP (SHapley Additive exPlanations) for model interpretability.

## Research Overview

### Objective
The primary objective of this research is to develop and evaluate advanced machine learning models for binary classification of water potability based on physicochemical parameters, while providing explainable insights into model predictions.

### Problem Statement
Access to safe drinking water is critical for public health. This study addresses the challenge of automatically classifying water samples as potable or non-potable using machine learning, enabling rapid and accurate water quality assessment.

## Methodology

### 1. Dataset
- **Source**: Water quality dataset with physicochemical parameters
- **Target Variable**: Potability (Binary: 0 = Non-potable, 1 = Potable)
- **Features**: Multiple numerical features including pH, hardness, solids, chloramines, sulfate, conductivity, organic carbon, trihalomethanes, and turbidity
- **Preprocessing**: Standardization using StandardScaler for feature normalization

### 2. Models Implemented

#### 2.1 Multi-Layer Perceptron (MLP)
- **Architecture**:
  - Input Layer → 128 neurons (BatchNorm + ReLU)
  - Hidden Layer 1 → 64 neurons (BatchNorm + ReLU + Dropout 0.3)
  - Hidden Layer 2 → 32 neurons (ReLU)
  - Output Layer → 2 neurons (Binary classification)
- **Training**: 40 epochs with Adam optimizer (LR: 0.001)
- **Use Case**: Baseline deep learning model for tabular data

#### 2.2 TabNet
- **Architecture**: Attention-based sequential attention mechanism
- **Key Features**:
  - Self-attention for feature selection
  - Sparse feature selection
  - Interpretable feature importance
  - Virtual batch size: 32
- **Training**: Up to 100 epochs with early stopping (patience: 15)
- **Use Case**: Specialized tabular data architecture with built-in interpretability

#### 2.3 Feature Tokenizer Transformer (FT-Transformer)
- **Architecture**:
  - Feature Tokenizer: Projects each numerical feature to d_token dimension
  - Transformer Encoder: 3 layers, 8 attention heads
  - Token dimension: 64
  - Feedforward dimension: 256
  - GELU activation with pre-layer normalization
- **Training**: 50 epochs with Adam optimizer (LR: 0.001)
- **Use Case**: Transformer-based architecture adapted for tabular data

### 3. Evaluation Framework

#### Cross-Validation Strategy
- **Method**: Stratified 5-Fold Cross-Validation
- **Purpose**: Robust performance estimation and generalization assessment
- **Metrics**:
  - Accuracy
  - F1-Score
  - ROC-AUC (Area Under the ROC Curve)
  - Precision and Recall
  - Confusion Matrix

#### Model Selection
- Best model selected based on mean ROC-AUC across all folds
- Final model retrained on entire dataset for deployment

### 4. Explainability Analysis

#### SHAP Integration
- **MLP & FT-Transformer**: GradientExplainer on GPU-accelerated computation
- **TabNet**: Model-agnostic Explainer with efficient sampling
- **Visualization**: Summary plots showing feature importance and contribution
- **Subset-Based Approach**: Uses 100-200 background samples for computational efficiency

### 5. Exploratory Data Analysis (EDA)

#### Performed Analyses:
1. **Target Distribution**: Class balance visualization
2. **Correlation Analysis**: Heatmap of feature correlations
3. **Feature Distribution**: KDE plots for top variance features stratified by potability
4. **Pairplot Analysis**: Multivariate relationships among top 4 features

## Experimental Setup

### Computational Environment
- **Device**: GPU-accelerated (CUDA) when available, CPU fallback
- **Framework**: PyTorch, PyTorch-TabNet
- **Random Seed**: 42 (for reproducibility)

### Hyperparameters
```python
BATCH_SIZE = 128
MLP_EPOCHS = 40
FTT_EPOCHS = 50
TABNET_MAX_EPOCHS = 100
LEARNING_RATE = 1e-3
N_SPLITS = 5 (Cross-validation folds)
RANDOM_SEED = 42
```

## Key Innovations

1. **Comprehensive Model Comparison**: First study to compare MLP, TabNet, and FT-Transformer on water potability prediction

2. **Unified Explainability Framework**: Integrated SHAP explanations for all three model types with GPU optimization

3. **Rigorous Validation**: 5-fold stratified cross-validation ensures robust performance estimates

4. **End-to-End Pipeline**: Complete workflow from EDA to model deployment with artifact saving

5. **Reproducibility**: Fixed random seeds and saved preprocessing artifacts (scaler, models)

## Results Structure

### Output Artifacts
The pipeline generates the following artifacts in the `artifacts/` directory:
- `scaler.pkl`: Fitted StandardScaler for preprocessing
- `mlp_full.pth`: Best MLP model weights (if selected)
- `ft_transformer_full.pth`: Best FT-Transformer weights (if selected)
- `tabnet_full.zip`: Best TabNet model (if selected)

### Performance Metrics
For each model and fold, the following are computed:
- Classification report (precision, recall, F1-score)
- Confusion matrix (raw and normalized)
- ROC curves with AUC scores
- Aggregated mean metrics across folds

### Visualizations Generated
1. Training loss curves per fold
2. Confusion matrices (absolute and normalized)
3. ROC curves (per-fold and aggregated)
4. SHAP summary plots for feature importance
5. EDA plots (distributions, correlations, pairplots)

## Installation and Dependencies

```bash
pip install pytorch-tabnet shap torch torchvision torchaudio
pip install scikit-learn matplotlib seaborn pandas numpy joblib
```

### Required Libraries
- **PyTorch**: Deep learning framework
- **PyTorch-TabNet**: TabNet implementation
- **SHAP**: Model explainability
- **scikit-learn**: Preprocessing, metrics, cross-validation
- **matplotlib, seaborn**: Visualization
- **pandas, numpy**: Data manipulation

## Usage

### Running the Complete Pipeline

```python
# 1. Place your dataset as 'water_quality_potability.csv'
# 2. Run all cells in sequence
# 3. Check 'artifacts/' directory for saved models
# 4. Review generated plots and metrics
```

### Loading Trained Models

```python
import joblib
import torch

# Load scaler
scaler = joblib.load('artifacts/scaler.pkl')

# Load MLP (example)
model = MLP(input_dim=9)  # Adjust based on your features
model.load_state_dict(torch.load('artifacts/mlp_full.pth'))
model.eval()

# Preprocess and predict
X_new_scaled = scaler.transform(X_new)
X_tensor = torch.tensor(X_new_scaled, dtype=torch.float32)
predictions = model(X_tensor).argmax(dim=1)
```

## Research Contributions

### Scientific Contributions
1. **Benchmarking Study**: Establishes performance baselines for water potability prediction using modern architectures
2. **Interpretability**: Demonstrates practical application of SHAP for environmental/health domain
3. **Methodology**: Provides replicable experimental framework for tabular classification tasks
4. **Feature Analysis**: Identifies key physicochemical parameters influencing water potability

### Practical Applications
- Automated water quality monitoring systems
- Real-time potability assessment in water treatment facilities
- Mobile water testing applications
- Environmental health surveillance

## Limitations and Future Work

### Current Limitations
1. SHAP computations use subsets (100-200 samples) for efficiency
2. Models assume numerical features only (no categorical handling demonstrated)
3. Class imbalance handling not explicitly addressed
4. Threshold optimization for decision-making not included

### Future Directions
1. **Ensemble Methods**: Combine predictions from all three models
2. **Hyperparameter Optimization**: Automated search (Optuna, Ray Tune)
3. **Temporal Analysis**: Incorporate time-series aspects if applicable
4. **Multi-class Extension**: Predict water quality grades beyond binary classification
5. **Deployment**: REST API for real-time inference
6. **Uncertainty Quantification**: Bayesian approaches for prediction confidence

## Paper Writing Guidelines

### Suggested Paper Structure

#### 1. Introduction
- Importance of water quality monitoring
- Limitations of traditional methods
- Motivation for machine learning approaches
- Research objectives and contributions

#### 2. Related Work
- Traditional water quality assessment methods
- Machine learning in environmental monitoring
- Deep learning for tabular data
- Explainable AI in healthcare/environmental domains

#### 3. Methodology
- Dataset description and preprocessing
- Model architectures (MLP, TabNet, FT-Transformer)
- Cross-validation strategy
- Evaluation metrics
- SHAP for explainability

#### 4. Experimental Setup
- Hardware and software specifications
- Hyperparameter settings
- Implementation details

#### 5. Results and Analysis
- Comparative performance metrics
- Statistical significance tests
- SHAP analysis and feature importance
- Case studies of predictions

#### 6. Discussion
- Interpretation of results
- Model strengths and weaknesses
- Practical implications
- Limitations

#### 7. Conclusion and Future Work
- Summary of contributions
- Future research directions

### Key Metrics to Report
- Mean ± Std of Accuracy, F1, AUC across folds
- Confusion matrices for best model
- Feature importance rankings
- Computational efficiency (training time, inference speed)

## Citation and References

If you use this code in your research, please cite relevant papers:
- TabNet: "TabNet: Attentive Interpretable Tabular Learning" (Arik & Pfister, 2021)
- FT-Transformer: "Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)
- SHAP: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)

## License
This research code is provided for academic and research purposes.

## Contact and Support
For questions regarding the implementation or methodology, please open an issue or contact the research team.

---

## Quick Start Checklist for Paper Writing

- [ ] Run complete pipeline and save all outputs
- [ ] Record mean ± std for all metrics across folds
- [ ] Save all generated plots in high resolution
- [ ] Document best model and its hyperparameters
- [ ] Analyze SHAP outputs for key findings
- [ ] Conduct statistical significance tests (t-tests, ANOVA)
- [ ] Prepare comparison tables (LaTeX format recommended)
- [ ] Identify 2-3 interesting case studies from predictions
- [ ] Review related literature for positioning your work
- [ ] Draft abstract highlighting novelty and contributions

## Notes for Reproducibility

1. **Dataset**: Ensure consistent train/test splits by using fixed random seed
2. **Environment**: Document exact package versions (`pip freeze > requirements.txt`)
3. **Hardware**: Note GPU model if using CUDA acceleration
4. **Runtime**: Record training time for each model for efficiency comparison
5. **Checkpoints**: Save model checkpoints at regular intervals for long training runs

---

**Last Updated**: October 2025
**Code Version**: 1.0
**Status**: Research Implementation
