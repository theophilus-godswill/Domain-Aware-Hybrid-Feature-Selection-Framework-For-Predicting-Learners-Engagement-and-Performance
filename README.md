# Domain-Aware Hybrid Feature Selection Framework (DAHFS)

**A domain-oriented hybrid feature selection framework for predicting learners' engagement and academic performance.**

**Researcher:** God'swill Theophilus  
**Research context:** PhD research  
**Implementation:** Python / Jupyter Notebook  
**Status:** Research prototype; associated paper submitted for review.

> DAHFS is designed for the learner engagement and academic performance prediction domain. It combines complementary feature-scoring strategies to address limitations that can arise when conventional feature-selection methods are applied individually to educational datasets.

---

## Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Proposed Framework](#proposed-framework)
- [Experimental Workflow](#experimental-workflow)
- [Predictive Model](#predictive-model)
- [Experimental Results](#experimental-results)
- [Repository Contents](#repository-contents)
- [Dataset](#dataset)
- [Installation and Setup](#installation-and-setup)
- [Running the Notebook](#running-the-notebook)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations and Responsible Use](#limitations-and-responsible-use)
- [Research and Citation](#research-and-citation)
- [Author](#author)

## Overview

Feature selection is an important stage in machine learning, particularly when datasets contain heterogeneous learner-related variables, overlapping information, and features that contribute differently to engagement and performance outcomes.

This project implements and evaluates a **Domain-Aware Hybrid Feature Selection Framework (DAHFS)** for learner engagement and academic performance prediction. The framework combines filter-based, embedded, and interaction-aware feature scoring, followed by score fusion, redundancy reduction, and bootstrap-based stability selection.

The selected features are subsequently used to train an LSTM-based classifier. The notebook also evaluates several conventional feature-selection techniques to provide a comparative experimental context.

The implementation is provided as an executable Jupyter Notebook so that researchers and practitioners can inspect the workflow, reproduce the experiments, and adapt the methods to related educational datasets.

## Research Motivation

Traditional feature-selection methods are often developed as general-purpose techniques. When applied individually to learner engagement and academic performance datasets, a single method may not capture all relevant characteristics of the educational domain.

For example, a method may prioritise individual feature relevance while overlooking feature redundancy, model-dependent importance, or relationships that are not adequately represented by one scoring strategy.

DAHFS is motivated by the need for a domain-oriented approach that combines complementary perspectives on feature relevance and selection stability. Its purpose is to manage some of the limitations of relying on a single traditional feature-selection technique when analysing learner engagement and performance data.

**Domain-aware** in this project refers to the framework's design and intended application for the learner engagement and academic performance prediction domain. It does not imply that an external educational ontology or manually curated expert-rule system is implemented.

## Proposed Framework

DAHFS integrates three feature-scoring components.

### 1. Filter-based feature scoring

The filter component evaluates feature relevance using:

- Mutual Information (MI)
- Chi-Squared (\(\chi^2\))
- ANOVA F-statistic

The three scores are combined to produce a filter-based score for each feature.

### 2. Embedded feature scoring

The embedded component combines feature importance obtained from:

- LASSO regression coefficients
- Random Forest feature importance

Absolute LASSO coefficients and Random Forest importance values are used to form the embedded score.

### 3. Interaction-aware feature scoring

The interaction-aware component combines:

- **PMI / discretised mutual-information component:** each feature is discretised using quantile-based bins, and its mutual information with the target is calculated.
- **HSIC:** a kernel-based dependence measure using RBF kernels to estimate feature–target dependence.

The implementation averages the PMI-related score and HSIC score to produce the interaction-aware score.

### 4. Hybrid score fusion

The implementation combines the three component scores using a weighted fusion strategy. The component scores are normalised before fusion so that they can contribute to a common ranking.

The framework's implementation should be consulted for the exact score normalisation and fusion operations when reproducing or extending the method.

### 5. Redundancy reduction and stability selection

The framework applies correlation-based pruning to reduce redundant features, using a correlation threshold of **0.85**.

It then performs bootstrap-based stability selection. Features are evaluated across repeated bootstrap samples, and their selection frequencies are used to identify consistently selected features.

The current notebook configuration uses:

| Parameter | Value |
|---|---:|
| Bootstrap iterations | 50 |
| Stability threshold (\(\tau\)) | 0.60 |
| Maximum final features | 20 |
| Random seed | 42 |
| Correlation pruning threshold | 0.85 |

The notebook reports the final stable feature set and visualises feature-selection frequencies and correlations among selected features.

### Framework workflow

```text
Learner engagement and performance dataset
                    |
                    v
           Data preprocessing
                    |
                    v
       +------------+-------------+
       |            |             |
       v            v             v
  Filter-based  Embedded     Interaction-aware
    scoring      scoring          scoring
       |            |             |
       +------------+-------------+
                    |
                    v
             Score normalisation
                    |
                    v
              Weighted fusion
                    |
                    v
          Correlation-based pruning
                    |
                    v
         Bootstrap stability selection
                    |
                    v
             Selected features
                    |
                    v
               LSTM model
                    |
                    v
       Performance evaluation and SHAP
```

## Experimental Workflow

The notebook contains the following experimental stages:

1. Load and inspect the learner dataset.
2. Explore and visualise the data.
3. Encode categorical variables and prepare numerical features.
4. Split the data into training and test subsets for the conventional comparison experiments.
5. Apply individual feature-selection techniques.
6. Train an LSTM classifier using the selected features from each technique.
7. Implement DAHFS and identify stable features.
8. Train and evaluate the LSTM model using the DAHFS-selected features.
9. Interpret model predictions using SHAP visualisations.

### Conventional feature-selection methods evaluated

The notebook includes experiments using:

- Mutual Information
- LASSO
- Recursive Feature Elimination (RFE)
- Information Gain
- Chi-Squared
- Random Forest feature importance
- Principal Component Analysis (PCA)

PCA is a dimensionality-reduction baseline and produces components rather than selecting original features in the same way as the feature-ranking methods.

## Predictive Model

The notebook uses a Long Short-Term Memory (LSTM) neural network as the downstream classifier.

For the tabular dataset, each observation is reshaped to a single time step before being passed to the LSTM. Accordingly, this implementation evaluates an LSTM architecture on tabular feature vectors; it does not model a longitudinal sequence of learner activity over multiple time steps.

The DAHFS experiment in the notebook uses:

- LSTM layer with 64 units
- Dropout rate of 0.3
- Dense hidden layer with 32 units and ReLU activation
- Adam optimiser
- 30 training epochs
- Batch size of 32
- 20% validation split from the training data

The notebook evaluates classification performance using accuracy, precision, recall, F1-score, classification reports, and confusion matrices.

## Experimental Results

### Conventional feature-selection comparison

The following values are transcribed from the comparative results table supplied with this project.

| Technique | Accuracy | Precision | Recall | F-measure |
|---|---:|---:|---:|---:|
| LSTM / Mutual Information | 90.5% | 90.5% | 90.5% | 90.5% |
| LSTM / LASSO | 82.4% | 83.1% | 82.4% | 82.4% |
| LSTM / RFE | 84.4% | 86.0% | 84.4% | 84.3% |
| LSTM / Random Forest | 96.6% | 96.5% | 96.5% | 96.5% |

### DAHFS-based experiment

The executed DAHFS-LSTM notebook output reports the following test metrics:

| Metric | Result |
|---|---:|
| Accuracy | 98.99% |
| Precision | 100.00% |
| Recall | 97.87% |
| F1-score | 98.92% |


## Repository Contents

The current research implementation is notebook-based.

```text
DAHFS/
│
├── FEATURE_SELECTION_UPDATED_ANALYSIS.ipynb
├── Student Engagement_Pilot_AUGMENTED.csv
├── README.md
└── requirements.txt
```

The dataset filename shown above is the filename referenced by the notebook. Add the dataset to the repository only after confirming that it is appropriate to share publicly and contains no identifying or restricted information.

## Dataset

The notebook loads:

```text
dataset
```

The target variable used in the notebook is:

```text
Class
```

The dataset includes learner engagement, assessment, demographic, and educational-context variables. The notebook uses categorical encoding and numerical scaling as part of preprocessing.

### Preparing the dataset

1. Obtain the dataset from the repository or the authorised dataset source.
2. Place the dataset in the same directory as the notebook.
3. Confirm that the target column is named `Class`.
4. Confirm that the dataset columns correspond to the variables referenced in the notebook.

If you use another dataset, you will need to adapt the target variable, preprocessing, and feature references accordingly.

## Installation and Setup

### Requirements

The implementation uses Python and the scientific Python and deep-learning ecosystem, including:

- Python
- Jupyter Notebook or JupyterLab
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow / Keras
- SHAP

A `requirements.txt` file can be created with the following starting dependencies:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
shap
jupyter
```

For reproducible research, record the exact package versions used for the final experiments and pin them in the requirements file.

### Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd YOUR-REPOSITORY
```

Replace the repository URL with the actual GitHub repository address.

### Create and activate a virtual environment

```bash
python -m venv dahfs-env
```

On Windows:

```bash
dahfs-env\Scripts\activate
```

On macOS / Linux:

```bash
source dahfs-env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `FEATURE_SELECTION_UPDATED_ANALYSIS.ipynb` and execute the notebook cells in order.

## Running the Notebook

Before execution:

- Ensure the dataset is available at the path expected by the notebook.
- Install all required dependencies.
- Review the data preprocessing and target encoding steps.
- Ensure the runtime has sufficient memory for the kernel-based HSIC calculations and neural-network training.

The notebook contains multiple experiments. Run the cells sequentially so that intermediate variables and selected feature sets are available to subsequent cells.

Google Colab can also be used. Upload the notebook and dataset, or mount the appropriate Drive location and update the dataset path.

## Reproducibility Notes

This repository represents a research implementation. Some aspects should be checked and standardised before treating the saved results as a fully reproducible benchmark.

1. **Randomness:** A seed of 42 is used in parts of the notebook. For strict reproducibility, seed Python, NumPy, and TensorFlow consistently and document the software and hardware environment.
2. **LSTM input interpretation:** Each tabular record is reshaped to one time step. This should not be interpreted as temporal sequence modelling of learner behaviour.

These considerations are included to make the experimental workflow transparent and to support future refinement of the research implementation.

## Limitations and Responsible Use

- The reported results are associated with the dataset and experimental configuration used in this research. Performance on other institutions, learner populations, or educational settings may differ.
- Predictive performance alone does not establish that a feature-selection framework is fair, unbiased, or suitable for consequential educational decisions.
- Learner data may contain sensitive information. Use appropriately authorised, de-identified data and follow applicable institutional and data-protection requirements.
- The framework is intended as a research contribution and should not be used as the sole basis for decisions about individual learners.

## Research and Citation

This implementation forms part of the author's PhD research. An associated paper has been submitted.

Full citation details can be added here when the paper is accepted or publicly available.

If you use or extend this implementation in academic work, please cite the associated publication when bibliographic details become available and acknowledge this repository.

## Author

**God'swill Theophilus**

- Google Scholar: https://scholar.google.com/citations?user=fu8zIFwAAAAJ&hl=en 
