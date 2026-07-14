# MVCHA-DP

Official implementation of **MVCHA-DP**, an adaptive multi-view hypergraph representation learning framework for multi-scenario software defect prediction.

This repository accompanies the paper:

> **Adaptive Multi-View Hypergraph Representation Learning for Multi-Scenario Software Defect Prediction**

## Overview

Software modules often exhibit complex dependencies and high-order collaborative relationships. However, existing hypergraph-based software defect prediction methods commonly rely on fixed neighborhood sizes, which may not adequately capture heterogeneous local structures. Moreover, distribution discrepancies across software versions and projects can reduce the transferability of learned representations.

MVCHA-DP addresses these challenges through adaptive multi-view hypergraph representation learning. It integrates multiple code feature views with learned node-importance priors and dynamically constructs hyperedges according to structural similarity, node importance, and local neighborhood characteristics.

The framework supports three representative software defect prediction scenarios:

- **WPDP:** Within-Project Defect Prediction
- **CVDP:** Cross-Version Defect Prediction
- **CPDP:** Cross-Project Defect Prediction

For WPDP, MVCHA-DP is mainly optimized through supervised classification. For CVDP and CPDP, cross-domain adaptation is additionally introduced to reduce distribution discrepancies between the source and target domains.

## Framework

The MVCHA-DP framework mainly consists of the following components:

1. Multi-view code feature generation
2. Node-importance computation
3. Adaptive multi-view hypergraph construction
4. Multi-view hypergraph fusion
5. Hypergraph convolutional representation learning
6. Scenario-specific optimization
7. Software defect classification

## Multi-View Code Features

MVCHA-DP uses three complementary feature views to represent software modules:

- **Traditional Static Metrics (TSM):** Code size, complexity, coupling, and object-oriented design metrics
- **Complex Network Metrics (CNM):** Structural indicators extracted from software dependency networks
- **Network Embedding Metrics (NEM):** Low-dimensional structural representations learned from dependency networks

These features are normalized and integrated to provide a comprehensive representation of each software module.

## Adaptive Hypergraph Construction

MVCHA-DP constructs adaptive hypergraphs by jointly considering:

- Structural similarity between software modules
- Learned node-importance priors
- Dynamic neighborhood selection
- Local similarity distributions
- Multi-view high-order relationships

Unlike fixed-neighborhood hypergraph construction, MVCHA-DP assigns different neighborhood sizes to different nodes according to their local structural characteristics.

The view-specific hypergraphs are subsequently fused and processed by a hypergraph convolutional network to learn high-order defect representations.

## Cross-Domain Adaptation

For CVDP and CPDP, MVCHA-DP incorporates a cross-domain adaptation mechanism consisting of:

- Adversarial domain alignment
- Gradient reversal learning
- Dual-classifier discrepancy regularization
- Source-domain supervised classification
- Target-domain boundary consistency learning

These components reduce representation discrepancies between source and target domains and improve prediction transferability across software versions and projects.

## Dataset

The experiments are conducted on multiple open-source Java projects from the **PROMISE software defect prediction benchmark dataset**.

The evaluated projects include:

- Ant
- Camel
- Ivy
- JEdit
- Log4j
- Lucene
- POI
- Velocity
- Xalan
- Xerces

Each project contains one or more software versions with file-level defect labels.

## Evaluation Scenarios

### Within-Project Defect Prediction

In WPDP, the training and testing samples are obtained from the same software project version. Five-fold cross-validation is used to evaluate the within-project prediction performance.

### Cross-Version Defect Prediction

In CVDP, an earlier version of a software project is used as the source domain, while a later version of the same project is used as the target domain.

### Cross-Project Defect Prediction

In CPDP, the source and target domains are obtained from different software projects.

## Evaluation Metrics

The following evaluation metrics are used:

- Area Under the ROC Curve (**AUC**)
- **F-measure**
- Matthews Correlation Coefficient (**MCC**)
- Geometric Mean (**G-mean**)

The Scott–Knott ESD test is also used to statistically compare different defect prediction methods.

## Repository Structure

The complete repository is expected to contain the following structure:

```text
MVCHA-DP/
├── configs/                 # Experiment configuration files
├── data/                    # Dataset preparation instructions
├── preprocessing/           # Feature extraction and preprocessing
├── models/                  # Model implementations
│   ├── hypergraph.py
│   ├── node_importance.py
│   ├── domain_adaptation.py
│   └── classifier.py
├── experiments/             # WPDP, CVDP, and CPDP experiments
├── utils/                   # Evaluation and utility functions
├── scripts/                 # Training and evaluation scripts
├── requirements.txt         # Python dependencies
└── README.md
