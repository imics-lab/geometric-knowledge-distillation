import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time
import os 

def extract_left_hand(sequence_data):
    """Extract left hand landmarks from the full sequence data.
    
    Assumes sequence_data has shape (window_size, num_channels),
    where num_channels = 225 and the right hand occupies 63 channels.
    """
    left_hand_start = 33*3 #54 * 3  # (33 pose + 21 left hand) * 3 coordinates
    left_hand_end = left_hand_start + 63  # 21 landmarks * 3 coordinates
    return sequence_data[:, left_hand_start:left_hand_end]

def modified_procrustes_distance_vectorized(X, Y, scale=True, sigma1=1.0, sigma2=1.0, sigma3=1.0, delta=0.0):
    """
    Vectorized modified Procrustes distance computation between one sequence and multiple references.
    Each frame is assumed to be represented as (n_landmarks, dim) with landmark 0 as the wrist.
    
    Parameters
    ----------
    X : torch.Tensor of shape (seq_len, n_landmarks, dim)
        Query sequence (e.g., shape (80, 21, 3)).
    Y : torch.Tensor of shape (n_refs, seq_len, n_landmarks, dim)
        Reference sequences.
    scale : bool
        Whether to use scaling in Procrustes alignment.
    sigma1, sigma2, sigma3 : float
        Normalization factors for alignment error, rotation penalty, and translation error.
    delta : float
        Translation threshold. Only translations larger than delta incur a penalty.
    
    Returns
    -------
    cost_matrices : torch.Tensor of shape (n_refs, seq_len, seq_len)
        For each reference and each pair of frames (query frame i, reference frame j),
        the normalized modified Procrustes distance.
    """
    device = X.device
    n_refs = Y.shape[0]
    seq_len = X.shape[0]
    n_landmarks = X.shape[1]  # expected 21
    dim = X.shape[2]          # expected 3
    
    # Preallocate cost matrices: one per reference, for each frame-to-frame pair.
    cost_matrices = torch.zeros((n_refs, seq_len, seq_len), device=device)
    
    # Loop over all frame indices in the query (i) and reference (j)
    # (The double loop is over sequence length only; the computations per (i,j) are batched over n_refs.)
    for i in range(seq_len):
        # Expand the query frame to shape (n_refs, n_landmarks, dim)
        X_frame = X[i].unsqueeze(0).expand(n_refs, -1, -1)  # shape: (n_refs, 21, 3)
        for j in range(seq_len):
            # Get the j-th frame from each reference sequence: shape (n_refs, 21, 3)
            Y_frame = Y[:, j, :, :]
            
            # --- Compute Translation (wrist difference) ---
            # Assume landmark 0 is the wrist.
            X_wrist = X_frame[:, 0, :]  # (n_refs, 3)
            Y_wrist = Y_frame[:, 0, :]  # (n_refs, 3)
            # f3: translation error is the Euclidean norm between wrists.
            f3 = torch.norm(X_wrist - Y_wrist, dim=1)  # shape: (n_refs,)
            
            # --- Compute Alignment Error (f1) and Rotation Penalty (f2) ---
            # Use the body landmarks (landmarks 1 to end) for shape alignment.
            X_body = X_frame[:, 1:, :]  # shape: (n_refs, 20, 3)
            Y_body = Y_frame[:, 1:, :]  # shape: (n_refs, 20, 3)
            
            # Center the body landmarks using the wrist as anchor.
            X0 = X_body - X_wrist.unsqueeze(1)  # shape: (n_refs, 20, 3)
            Y0 = Y_body - Y_wrist.unsqueeze(1)  # shape: (n_refs, 20, 3)
            
            # Optionally apply scale normalization.
            if scale:
                # Compute the Frobenius norm per sample.
                X_norm = torch.norm(X0.reshape(n_refs, -1), p='fro', dim=1, keepdim=True)  # (n_refs, 1)
                Y_norm = torch.norm(Y0.reshape(n_refs, -1), p='fro', dim=1, keepdim=True)
                X0 = X0 / (X_norm.view(n_refs, 1, 1) + 1e-15)
                Y0 = Y0 / (Y_norm.view(n_refs, 1, 1) + 1e-15)
            
            # Compute the optimal rotation matrix R for each sample.
            # A = X0^T @ Y0 for each sample.
            A = torch.bmm(X0.transpose(1, 2), Y0)  # shape: (n_refs, 3, 3)
            # Compute SVD for each sample.
            U, S, Vh = torch.linalg.svd(A)  # U: (n_refs, 3, 3), Vh: (n_refs, 3, 3)
            # Compute R = U @ Vh.
            R = torch.bmm(U, Vh)  # shape: (n_refs, 3, 3)
            
            # Adjust for reflections: if det(R) < 0, flip sign on last row of Vh.
            det_R = torch.linalg.det(R)  # shape: (n_refs,)
            if (det_R < 0).any():
                Vh_adjusted = Vh.clone()
                mask = (det_R < 0)
                Vh_adjusted[mask, -1, :] *= -1
                R = torch.bmm(U, Vh_adjusted)
            
            # Rotation penalty: compute the rotation angle theta (in radians) for each sample.
            # For a rotation matrix in 3D, theta = arccos((trace(R)-1)/2).
            trace_R = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]  # shape: (n_refs,)
            cos_theta = (trace_R - 1) / 2
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            theta = torch.acos(cos_theta)
            f2 = theta ** 2  # squared rotation penalty, shape: (n_refs,)
            
            # Rotate Y0 to obtain the aligned configuration.
            Y0_rot = torch.bmm(Y0, R)  # shape: (n_refs, 20, 3)
            diff = X0 - Y0_rot
            # Alignment error: Frobenius norm over landmarks and dimensions.
            f1 = torch.norm(diff.reshape(n_refs, -1), dim=1)  # shape: (n_refs,)
            
            # --- Combine the components using the empirical scales and threshold ---
            # For translation, penalize only the excess over delta.
            translation_penalty = (torch.clamp(f3 - delta, min=0) ** 2)
            # Final per–frame cost:
            cost = f1 / sigma1 + f2 / sigma2 + translation_penalty / sigma3  # shape: (n_refs,)
            
            cost_matrices[:, i, j] = cost
    return cost_matrices

def dtw_vectorized(C):
    """
    Vectorized DTW computation for a batch of cost matrices.
    
    Parameters
    ----------
    C : torch.Tensor of shape (batch_size, seq_len, seq_len)
        Batch of cost matrices.
        
    Returns
    -------
    dtw_costs : torch.Tensor of shape (batch_size,)
        DTW distances for each cost matrix.
    """
    batch_size, m, n = C.shape
    D = torch.full((batch_size, m+1, n+1), float('inf'), device=C.device)
    D[:, 0, 0] = 0.0

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = C[:, i-1, j-1]
            # Take the minimum over the three neighboring cells.
            prev_min = torch.min(torch.min(D[:, i-1, j], D[:, i, j-1]), D[:, i-1, j-1])
            D[:, i, j] = cost + prev_min

    return D[:, -1, -1]

def compute_knn_for_one_instance(query, references, k, sigma1, sigma2, sigma3, delta):
    """
    Compute k-nearest neighbors for a single query sequence using the modified Procrustes-DTW.
    
    Parameters
    ----------
    query : torch.Tensor of shape (80, 225)
        Single query sequence.
    references : torch.Tensor of shape (n_refs, 80, 225)
        Reference sequences.
    k : int
        Number of nearest neighbors to find.
    sigma1, sigma2, sigma3, delta : float
        Empirical normalization parameters and translation threshold.
        
    Returns
    -------
    distances : torch.Tensor of shape (k,)
        DTW distances to the k-nearest neighbors.
    indices : torch.Tensor of shape (k,)
        Indices of the k-nearest neighbors.
    """
    # Extract right-hand landmarks from the query and each reference.
    query_hand = extract_left_hand(query)  # shape: (80, 63)
    ref_hands = torch.stack([extract_left_hand(ref) for ref in references])  # shape: (n_refs, 80, 63)
    
    # Reshape to (seq_len, n_landmarks, dim) where n_landmarks=21, dim=3.
    query_hand = query_hand.reshape(80, 21, 3)
    ref_hands = ref_hands.reshape(-1, 80, 21, 3)
    
    # Compute vectorized cost matrices using the modified Procrustes distance.
    cost_matrices = modified_procrustes_distance_vectorized(query_hand, ref_hands,
                                                            scale=True,
                                                            sigma1=sigma1, sigma2=sigma2,
                                                            sigma3=sigma3, delta=delta)
    # Compute DTW distances (one per reference).
    dtw_distances = dtw_vectorized(cost_matrices)
    
    # Find the k smallest distances (nearest neighbors).
    distances, indices = torch.topk(dtw_distances, k, largest=False)
    return dtw_distances, indices

def compute_knn_all(test_sequences, train_sequences, k, sigma1, sigma2, sigma3, delta):
    """
    Compute k-nearest neighbors for all test sequences using the modified Procrustes-DTW distance.
    
    Parameters
    ----------
    test_sequences : torch.Tensor of shape (n_test, 80, 225)
        Test sequences.
    train_sequences : torch.Tensor of shape (n_train, 80, 225)
        Training sequences.
    k : int
        Number of nearest neighbors to find.
    sigma1, sigma2, sigma3, delta : float
        Empirical normalization parameters and translation threshold.
        
    Returns
    -------
    all_distances : torch.Tensor of shape (n_test, k)
        DTW distances to the k-nearest neighbors.
    all_indices : torch.Tensor of shape (n_test, k)
        Indices of the k-nearest neighbors.
    """
    n_test = test_sequences.shape[0]
    n_train = train_sequences.shape[0]
    device = torch.device('cuda')
    
    all_distances = torch.zeros(n_test, n_train, device=device)
    all_indices = torch.zeros(n_test, k, dtype=torch.long, device=device)
    
    for i in tqdm(range(0, 1000), desc="Processing test sequences"):
        torch.cuda.empty_cache()  # Optional: free memory between iterations
        distances, indices = compute_knn_for_one_instance(test_sequences[i],
                                                          train_sequences, k,
                                                          sigma1, sigma2, sigma3, delta)
        all_distances[i] = distances
        all_indices[i] = indices
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_test} test sequences; GPU memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
    
    return all_distances, all_indices

if __name__=="__main__":
    total_start_time = time.time()
    X_train = np.load('data_numpy/X_train.npy')
    X_test = np.load('data_numpy/X_test.npy')
    train_X = torch.tensor(X_train) #torch.tensor(train_X) if not isinstance(train_X, torch.Tensor) else train_X
    test_X = torch.tensor(X_test)

    # Move to GPU
    print("Moving data to GPU...")
    train_X = train_X.cuda()
    test_X = test_X.cuda()

    print(f"\nInput shapes:")
    print(f"Train data: {train_X.shape}")
    print(f"Test data: {test_X.shape}")
    
    sigma1 = 0.5110750971721483
    sigma2 = 1.6964131956660402
    sigma3 = 0.1190637305732086
    delta  = 0.20439801800608812
    k = 30
    print(f"\nComputing {k}-NN for all test sequences...")
    distances, indices = compute_knn_all(test_X, train_X, k, sigma1, sigma2, sigma3, delta)
    print(f"\nResults:")
    print(f"Distances shape: {distances.shape}")
    print(f"Indices shape: {indices.shape}")

    total_time = time.time() - total_start_time
    print(f"\nComputation completed!")
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per test sequence: {total_time/len(test_X):.2f} seconds")

    np_distances = distances.cpu().numpy()
    np_indices = indices.cpu().numpy()
    np.save('results_new_test_left/distances_left_1400_1600.npy', np_distances)

    # output_dir = './outputs'
    # os.makedirs(output_dir, exist_ok=True)
    # np.save(os.path.join(output_dir,'distances_left_0_750.npy'), np_distances)

        
