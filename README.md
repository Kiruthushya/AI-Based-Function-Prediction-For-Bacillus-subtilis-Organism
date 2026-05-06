# AI-Based-Function-Prediction-For-Bacillus-subtilis-Organism
AI-based multi-label gene ontology prediction using ESM2 and ESMFold for functional annotation of hypothetical proteins.

## Overview

This project presents an AI-driven framework for predicting Gene Ontology (GO) terms of hypothetical proteins using transformer-based protein language models and structural bioinformatics.

The framework integrates:

- ESM-2 protein sequence embeddings
- ESMFold structure prediction
- Structural feature extraction
- Feature fusion
- Deep neural network-based multi-label classification

The primary objective of this work is to improve functional annotation of hypothetical proteins without relying solely on sequence homology.

---

# Problem Statement

A significant proportion of proteins in Bacillus subtilis remain annotated as hypothetical proteins, limiting biological interpretation and downstream functional studies.

Traditional sequence similarity and domain-based annotation approaches often fail to assign functions to proteins with low homology.

This project addresses the problem using deep learning-based sequence representations and structure-aware feature fusion for multi-label Gene Ontology prediction.

---

The overall workflow consists of:

1. Protein sequence collection from UniProt
2. GO annotation extraction
3. Matrix label preparation
4. ESM-2 embedding generation
5. ESMFold structure prediction
6. Structural feature extraction
7. Feature fusion
8. Fusion neural network training
9. External validation and evaluation

---

# Dataset Information

Protein sequences and Gene Ontology annotations were obtained from UniProt.

## Bacillus subtilis strain 168

- Total protein sequences: 8662
- Hypothetical proteins removed: 1809
- Final annotated proteins used for training: 6853

---

# Sequence Embedding Generation

Protein sequences were passed through the pretrained ESM-2 model to generate sequence embeddings.

## Embedding Details

- Model: ESM2
- Embedding dimension: 1280
- Mean pooling applied to obtain fixed-length protein representation

Generated embeddings were stored as:

```python
sequence_embeddings.npy
```

---

# Structure Prediction and Feature Extraction

Protein 3D structures were predicted using ESMFold.

## Structural Features Extracted

- Protein length
- Mean residue distance
- Standard deviation of residue distances
- Hydrophobic residue ratio
- Polar residue ratio
- Charged residue ratio

Extracted features were stored as:

```python
structure_features.npy
```

---

# Feature Fusion

Sequence embeddings (1280 features) were concatenated with structural features (6 features).

## Final Feature Vector

```python
1280 + 6 = 1286 features
```

Combined features were stored as:

```python
combined_features.npy
```

---

# Model Architecture

The project uses a Fusion Neural Network for multi-label GO term prediction.

## Architecture

- Input Layer → 1286 neurons
- Hidden Layer 1 → 512 neurons + ReLU + Dropout
- Hidden Layer 2 → 256 neurons + ReLU + Dropout
- Output Layer → 100 GO terms
- Sigmoid activation for multi-label classification

## Loss Function

```python
BCEWithLogitsLoss
```

## Optimizer

```python
AdamW
```

---

# Model Performance

| Metric | Score |
|---|---|
| Micro ROC-AUC | 96.15% |
| Exact GO Match | 63.95% |
| Functional Group Match | 85.71% |

---

# External Validation

External validation was performed using Bacillus subtilis PY79 strain.

Validation strategies included:

- Exact GO term matching
- Functional group matching

The model demonstrated strong biological relevance and effective generalization on unseen protein sequences.

---

# Advantages of the Proposed Model

- Does not rely solely on sequence similarity
- Captures evolutionary and biochemical patterns
- Integrates structural information
- Improves prediction performance significantly
- Scalable for large-scale protein annotation

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- BioPython
- Scikit-learn
- Matplotlib
- ESM-2
- ESMFold

---
