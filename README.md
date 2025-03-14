# Geometric Knowledge Distillation

## Geometric Knowledge Distillation

This project implements knowledge distillation techniques using geometric transformations, focusing on three main models for each dataset: **KNN**, **Transformer**, and **Distillation**. The experiments are carried out on two datasets: **Skeleton** and **Signum**. The objective is to evaluate how each model performs on different geometric transformation tasks.

## Description

This repository provides an implementation of geometric knowledge distillation for improving machine learning model performance through transformation-based techniques. We explore and compare different methods such as KNN, Transformers, and Distillation models on two datasets: **Skeleton** and **Signum**. The experiments are designed to evaluate performance on both synthetic and real-world datasets.

## Setup

### Conda Virtual Environment

Below are some base instructions for setting up the environment. For best practices, please use `conda` to manage your environments.

1. Create the Conda virtual environment using the `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the environment:

    ```bash
    conda activate YOUR_PROJECT_NAME
    ```

3. Set the Python path dynamically for the environment:

    ```bash
    conda env config vars set PYTHONPATH=$(pwd):$(pwd)/src
    ```

4. Verify that the environment has been created and activated:

    ```bash
    conda info --envs
    ```

### Dependencies

This project requires the following dependencies:

- Python >= 3.7
- `conda`
- `PyTorch`
- `scikit-learn`
- `transformers`
- `matplotlib`
- `numpy`
- `pandas`
- `tqdm`





<!-- CITATION -->
## Citation



```bibtex
@article{YourName,
  title={Your Title},
  author={Your team},
  year={Year}
}
```

