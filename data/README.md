# Data

We use two dataset in this project.

## Signum Dataset
The [SIGNUM dataset](https://www.phonetik.uni-muenchen.de/Bas/SIGNUM/) contains 450 basic signs (*classes*) from German Sign Language (DGS), performed by 25 different signers. We extract 3D hand landmarks (21 points per hand) using MediaPipe Holistic [mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide), resulting in a 126-dimensional feature vector per frame (21 landmarks × 3 coordinates × 2 hands). The sequence length is 80 timesteps. Following standard practice for signer-independent evaluation, we split the dataset by signer ID:

- **Training set**: 14 signers (IDs 1-14), 6,300 sequences
- **Validation set**: 4 signers (IDs 15-18), 1,800 sequences
- **Test set**: 7 signers (IDs 19-25), 1,901 sequences

## UTH-MHAD Dataset
The [UTD-MHAD dataset](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) comprises 27 actions performed by 8 subjects, with each action repeated 4 times. We use the 3D skeleton data captured by a Kinect sensor, which provides 20 joint positions in 3D space. The sequence length varies from 45 to 125 timesteps. The dataset contains 861 sequences after removing corrupted samples. For subject-independent evaluation, we use:

- **Training set**: 5 subjects (IDs 1-5), 539 sequences
- **Test set**: 3 subjects (IDs 6-8), 322 sequences

Due to the small number of subjects in this dataset, we do not use a validation set, as extensive hyperparameter tuning is not the objective of this study.

Due to space limitations only MHAD Dataset is kept in github and uploaded as numpy files. Since the MHAD dataset has variable window size, each sample has been padded to maximum length of 125 and the original length is stored in file org_train.npy and org_test.npy for train and test set respectively.
