# Geometric Knowledge Distillation

## Overview
Motion sequence classification using geometric approaches like Procrustes analysis demonstrates high accuracy but suffers from computational inefficiency at inference time. We present a novel knowledge distillation framework that bridges this gap by transferring geometric understanding from Procrustes combined with Dynamic Time Warping (Procrustes-DTW) distance computations to an efficient neural network. Our approach uses pre-computed Procrustes-DTW distances to generate soft probability distributions that guide the training of a transformer-based student model. This ensures the preservation of crucial geometric properties—including shape similarities, temporal alignments, and invariance to spatial transformations—while enabling fast inference. We evaluate our framework on two challenging tasks: sign language recognition using the SIGNUM dataset and human action recognition using the UTD-MHAD dataset. Experimental results demonstrate that geometric knowledge transfer improves accuracy compared to training a deep neural network using standard supervised learning while achieving significantly faster inference times compared to distance-based approaches. The framework shows particular promise for real-time applications where both geometric understanding and computational efficiency are essential.


This repository implements **Geometric Knowledge Distillation**, applying transformation-based knowledge distillation techniques to improve machine learning model performance. The project focuses on three primary models:
- **KNN (k-Nearest Neighbors)**
- **Transformer-based models**
- **Distillation models**

The experiments are conducted on two datasets:
- **Skeleton Dataset** (included in the repository)
- **Signum Dataset** (not included due to size limitations)

## Repository Structure
```
.github/workflows/      # CI/CD workflows (GitHub Actions)
data/                   # Skeleton dataset (Signum dataset not included due to size)
docs/                   # Documentation files
results/                # Output results from experiments
scripts/                # Scripts for running different algorithms
tests/                  # Unit tests for verifying implementations
src/                    # Core source files for geometric knowledge distillation
.amlignore              # Azure ML ignore file (similar to .gitignore)
.gitignore              # Files ignored by Git
.pre-commit-config.yaml # Pre-commit hooks configuration
CITATION.cff            # Citation information
LICENSE                 # License details
README.md               # This file
environment.yml         # Conda environment setup file
setup.py                # Project setup file
```

## Scripts
The `scripts/` directory contains implementations for different algorithms used in the project:
```
scripts/
│── knn_signum.py                  # KNN model for Signum dataset
│── knn_skeleton_git.py             # KNN model for Skeleton dataset
│── signum_distillation.ipynb       # Distillation model for Signum dataset
│── signum_transformer.ipynb        # Transformer model for Signum dataset
│── skeleton_distillation.ipynb     # Distillation model for Skeleton dataset
│── skeleton_procrustes.py          # Procrustes analysis on Skeleton dataset
│── skeleton_transformer.ipynb      # Transformer model for Skeleton dataset
```

## Setup
### Conda Virtual Environment
To set up the environment, follow these steps:
1. Create the Conda virtual environment using `environment.yml`:
    ```bash
    conda env create -f environment.yml
    ```
2. Activate the environment:
    ```bash
    conda activate distillation
    ```
3. Set the Python path dynamically:
    ```bash
    conda env config vars set PYTHONPATH=$(pwd):$(pwd)/src
    ```
4. Verify the environment setup:
    ```bash
    conda info --envs
    ```

### Dependencies
This project requires the following dependencies:
- Python >= 3.7
- `PyTorch`
- `scikit-learn`
- `transformers`
- `matplotlib`
- `numpy`
- `pandas`
- `tqdm`

## Data Availability
- **Skeleton Dataset** is included in `data/`
- **Skeleton Results** are available in `results/`
- **Signum Dataset & Results** are not included due to size limitations

## Running Experiments
To run the models on the datasets:
```bash
python scripts/skeleton_procrustes.py  # Runs the skeleton dataset experiments
python scripts/knn_signum.py           # Runs KNN on Signum dataset
python scripts/knn_skeleton_git.py     # Runs KNN on Skeleton dataset
```
For other experiments, refer to the `.ipynb` notebooks in the `scripts/` directory.

<!---
## Citation
If you use this work, please cite:
```bibtex
@article{YourName,
  title={Geometric Knowledge Distillation},
  author={Your team},
  year={Year}
}
```
--->

## License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.

