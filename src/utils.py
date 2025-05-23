import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

def compute_teacher_probabilities(dtw_distances: torch.Tensor, 
                                train_labels: torch.Tensor,
                                num_classes: int, 
                                temperature: float = 1.0) -> torch.Tensor:
    """
    Compute soft probability distributions from DTW distances
    Args:
        dtw_distances: tensor of shape (batch_size, num_training_examples)
        train_labels: tensor of shape (num_training_examples) with class labels of all training examples
        num_classes: total number of classes
        temperature: temperature parameter for softening distributions
    Returns:
        soft probabilities of shape (batch_size, num_classes)
    """
    # Convert distances to similarities (negative distances)
    similarities = -dtw_distances / temperature
    # Apply softmax to get example-wise probabilities
    example_probs = F.softmax(similarities, dim=1)  # (batch_size, num_training_examples)
    # Initialize class probabilities tensor
    batch_size = dtw_distances.shape[0]
    class_probs = torch.zeros(batch_size, num_classes, device=dtw_distances.device)
    # For each class, sum the probabilities of examples belonging to that class
    for i in range(num_classes):
        class_indices = (train_labels == i).nonzero(as_tuple=True)[0]
        if len(class_indices) > 0:
            # Sum (not mean) probabilities of all examples of this class
            class_probs[:, i] = example_probs[:, class_indices].sum(dim=1)
    # Re-normalize to ensure proper probability distribution
    # This is necessary because we're summing probabilities across examples
    class_probs = class_probs / (class_probs.sum(dim=1, keepdim=True) + 1e-8)
    return class_probs
