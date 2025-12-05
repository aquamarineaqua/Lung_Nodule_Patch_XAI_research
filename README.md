# LIDC-IDRI Patch Research

An Interpretable Concept-based Approach for Pulmonary Nodule Malignancy Prediction using ConceptCLIP

## Introduction

This project explores the use of **ConceptCLIP** (a vision-language model tailored for the biomedical domain) for pulmonary nodule benign/malignant classification. The core approach includes:

1. **Zero-shot Feature Extraction**: Using a pre-trained ConceptCLIP model without fine-tuning
2. **Concept-driven**: Defining radiological concepts (e.g., spiculation, lobulation, calcification) and computing image-concept similarity scores
3. **Interpretability**: Using simple logistic regression models to quantify each concept's contribution through feature coefficients

### Research Questions

- **RQ1**: How to design effective radiological concepts to improve downstream classification performance?
- **RQ2**: Are the concept representations learned by ConceptCLIP clinically interpretable?

### Main Results

- 20-concept configuration achieves optimal performance: **AUC=0.794, Accuracy=0.730**
- **12 statistically significant concept features** identified through L1-regularized feature selection
- All significant features align with established clinical knowledge

## Dataset

Nodule-level 2D patch dataset constructed from **LIDC-IDRI**:

- **Number of Nodules**: 678 (Malignant: 397, Benign: 281)
- **Number of Patients**: 440
- **Number of Images**: 2532 patches

Dataset construction code: [LIDC-IDRI_patch_generation](https://github.com/aquamarineaqua/LIDC-IDRI_patch_generation)

## Project Structure

```
LIDC-IDRI_patch_research/
├── 1_Database_and_Dataloader_create.ipynb  # Data preprocessing and feature extraction
├── 2_Read_database.ipynb                    # Database reading and basic classification
├── 3_For_20_concept.ipynb                   # 20-concept experiment
├── 4_For_30_concept.ipynb                   # 30-concept experiment
├── 5_For_image_embeddings.ipynb             # Image embedding baseline experiment
├── 6_For_20_concept_L1_feature_selection.ipynb  # L1 feature selection and interpretability analysis
└── datasets/                                # Dataset folder
    └── curation2/lidc_patches_all/          # Patch image data (please extract the compressed file)
```

## Installation

### Python Version
- Python >= 3.10

### Required Libraries

```bash
pip install torch torchvision
pip install transformers
pip install huggingface_hub
pip install pandas numpy
pip install matplotlib
pip install scikit-learn
pip install statsmodels
pip install h5py
pip install tqdm
pip install pillow
pip install seaborn
pip install jupyter
```

Or install all at once:

```bash
pip install torch torchvision transformers huggingface_hub pandas numpy matplotlib scikit-learn statsmodels h5py tqdm pillow seaborn jupyter
```

## Usage Guide

### 1. Data Preparation

Ensure the `datasets/curation2/lidc_patches_all/` directory contains:
- `all_patches_metadata.csv`: Metadata file
- Patient folders (e.g., `LIDC-IDRI-0001/`): Containing patch images

**Note**: First-time users need to log in to HuggingFace:
```python
from huggingface_hub import login
login(token="YOUR_HUGGINGFACE_TOKEN")
```

### 2. Notebook Descriptions

| Notebook | Description |
|----------|-------------|
| `1_Database_and_Dataloader_create.ipynb` | **Data Preprocessing & Feature Extraction Pipeline**: Filter nodules with area ≥50mm², generate binary labels, load ConceptCLIP to extract image/text features, store in HDF5 database |
| `2_Read_database.ipynb` | **Basic Classification Experiment (10 concepts)**: Read HDF5 features, Prompt Ensembling, compute concept scores, nodule-level aggregation, 5-fold cross-validation logistic regression |
| `3_For_20_concept.ipynb` | **20-Concept Experiment**: Define 20 radiological concept categories (98 sub-concepts), generate text embeddings, image-concept similarity computation, classification evaluation |
| `4_For_30_concept.ipynb` | **30-Concept Experiment**: Expand to 30 concept categories (145 sub-concepts), compare classification performance across different concept counts |
| `5_For_image_embeddings.ipynb` | **Baseline Experiment**: Classification using pure image embeddings (Mean-Max concatenation, 2304 dimensions), comparing Logistic Regression, SVM, and XGBoost |
| `6_For_20_concept_L1_feature_selection.ipynb` | **Feature Selection & Interpretability Analysis**: LASSO feature selection, statistical significance testing (p-value), Bootstrap stability evaluation, Case Study visualization |

### 3. Recommended Workflow

```
Step 1: Data Preprocessing
└── 1_Database_and_Dataloader_create.ipynb
    ├── Output: curated_metadata.csv
    └── Output: conceptclip_features.h5

Step 2: Concept Experiments
├── 2_Read_database.ipynb (10 concepts)
├── 4_For_30_concept.ipynb
└── 3_For_20_concept.ipynb

Step 3: Interpretability Analysis
└── 6_For_20_concept_L1_feature_selection.ipynb
    ├── LASSO Feature Selection
    ├── Statistical Testing
    └── Case Study Visualization

(Optional) Baseline Comparison
└── 5_For_image_embeddings.ipynb
```

### 4. Output Files

| File | Description |
|------|-------------|
| `curated_metadata.csv` | Filtered metadata |
| `conceptclip_features.h5` | Original 10-concept feature database |
| `conceptclip_features_20.h5` | 20-concept feature database |
| `conceptclip_features_30.h5` | 30-concept feature database |
| `image_features/df_image_features.csv` | Image embedding features |
| `image_features/df_nodule_features_concept20_minmax.csv` | Nodule-level concept scores |

## Concept Design

This project defines 20 radiological concept categories (examples):

| Concept Category | Clinical Significance |
|------------------|----------------------|
| `spiculation` | Spiculated margins - Malignancy indicator |
| `lobulation` | Lobulation - Malignancy indicator |
| `round_shape` | Round shape - Benign tendency |
| `solid` | Solid nodule |
| `part_solid` | Part-solid - High malignancy risk |
| `benign_calc` | Benign calcification |
| `cavity` | Cavitation |

Each concept contains multiple synonym variants, combined with 10 prompt templates to generate text descriptions.

## License

MIT License
