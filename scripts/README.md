# Scripts

The following Python files and Jupyter notebook demos are provided for running experiments on the MHAD Dataset. Below are the steps for performing various tasks:

### 1. Compute the Procrustes DTW distance for each **train-train** sample:
Run the `skeleton_procrustes.py` using the training set.

```bash
python skeleton_procrustes.py 
```
### 2. Compute the Procrustes DTW distance for each train-test pair:
Run the skeleton_procrustes.py using the test set.

```bash
python skeleton_procrustes.py
```
### 3. K-Nearest Neighbors (KNN):
To run the KNN algorithm on the dataset, use the knn_skeleton.py script.

```bash

python knn_skeleton.py
```
### 4. Transformer Model:
Run the skeleton_transformer.ipynb notebook for transformer-based model training and evaluation.

```bash

jupyter notebook skeleton_transformer.ipynb
```
### 5. Distillation:
To perform distillation, run the skeleton_distillation.ipynb notebook.

```bash

jupyter notebook skeleton_distillation.ipynb
```
By following these steps and running the scripts, you will be able to perform all the experiments on the MHAD dataset.
