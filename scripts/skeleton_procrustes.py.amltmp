import numpy as np
import time
from tqdm import tqdm

def procrustes_distance(X, Y, scale=True):
    """
    Procrustes distance between two point sets X and Y.

    Parameters
    ----------
    X : ndarray of shape (n_points, dim)
        The first shape's landmarks.
    Y : ndarray of shape (n_points, dim)
        The second shape's landmarks.
    scale : bool, optional
        Ues scale normalization in the alignment [Arvanitis et.al]

    Returns
    -------
    dist : float
        The Procrustes distance between X and Y
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # 1. Translation (centroid to origin)
    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    X0 = X - X_mean
    Y0 = Y - Y_mean

    # 2. Scale
    normX = np.linalg.norm(X0)
    normY = np.linalg.norm(Y0)
    if scale:
        X0 /= (normX + 1e-15)
        Y0 /= (normY + 1e-15)

    # 3. Find best rotation
    A = X0.T @ Y0
    U, s, Vt = np.linalg.svd(A)
    R = U @ Vt

    # Reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # 4. Rotate
    Y0_rot = Y0 @ R

    # 5. Procrustes distance
    diff = X0 - Y0_rot
    dist = np.sum(np.linalg.norm(diff, axis=1))

    return dist

def dtw_procrustes(seqA, seqB, n_landmarks=75, dim=3, scale=True):
    """
    Procrustes-DTW distance between two sequences of landmarks.

    Parameters
    ----------
    seqA : ndarray of shape (m, no_landmarks * dim)
        Each frame is a flat array of landmarks.
    seqB : ndarray of shape (n, no_landmarks * dim)
        The second sequence of frames.
    n_landmarks : int, optional
        Number of landmarks per frame
    dim : int, optional
        Dimensionality of each landmark
    scale : bool, optional
        Whether to include scaling normalization in the Procrustes alignment

    Returns
    -------
    dtw_cost : float
        Procrustes-DTW alignment cost.
    """

    m = len(seqA)
    n = len(seqB)

    expected_no_landmarks = n_landmarks * dim
    if seqA.shape[1] != expected_no_landmarks or seqB.shape[1] != expected_no_landmarks:
        raise ValueError(f"Each frame must have {expected_no_landmarks} values (n_landmarks * dim).")

    # Reshape sequences: (m, no_landmarks * dim) -> (m, n_landmarks, dim)
    seqA_reshaped = seqA.reshape(m, n_landmarks, dim)
    seqB_reshaped = seqB.reshape(n, n_landmarks, dim)

    # cost matrix
    C = np.zeros((m, n), dtype=float)

    for i in range(m):
        X_i = seqA_reshaped[i]
        for j in range(n):
            Y_j = seqB_reshaped[j]
            C[i, j] = procrustes_distance(X_i, Y_j, scale=scale)

    # DTW matrix
    D = np.full((m+1, n+1), np.inf, dtype=float)
    D[0,0] = 0.0

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost_ij = C[i-1, j-1]
            D[i, j] = cost_ij + min(
                D[i-1, j],
                D[i, j-1],
                D[i-1, j-1]
            )

    return D[m, n]

def remove_padding(padded_data, original_length=60):
    # Assuming that the original length is known (e.g., 60)
    return padded_data[:original_length,:]




# --- Example Usage ---
if __name__ == "__main__":
    # Assume train_data is a numpy array of shape (6300, 80, 225)
    # and train_labels is a numpy array of shape (6300,)
     
    X_train = np.load('data/Skeleton_numpy/X_train.npy')
    X_test = np.load('data/Skeleton_numpy/X_test.npy')
    org_train = np.load('data/Skeleton_numpy/org_train.npy')
    org_test = np.load('dtaa/Skeleton_numpy/org_test.npy')


    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])
    n_test = X_test.shape[0]
    n_train = X_train.shape[0]

    all_distances = np.zeros((n_test, n_train))
    for test_i in tqdm(range(0, n_test), desc="Processing test sequences"):
        start = time.time()
        for train_j in range(n_train):
            test_seq = X_test[test_i]
            train_seq = X_train[train_j]

            org_train_j = org_train[train_j]
            org_test_i = org_test[test_i]
            
            clean_train = remove_padding(train_seq, org_train_j)
            clean_test = remove_padding(test_seq, org_test_i)
            dtw_cost = dtw_procrustes(clean_test, clean_train, n_landmarks=20, dim=3, scale=True)                               
            
        #   print("DTW cost =", dtw_cost) 
            all_distances[test_i][train_j] = dtw_cost
        end = time.time()
        print('Time: ', end-start)

    np.save('results/skeleton/distances_train.npy', all_distances)



    
    
