import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import matplotlib.pyplot as plt
import random

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def extract_both_hand(sequence_data):
    """Extract right hand landmarks from the full sequence data.
   
    Assumes sequence_data has shape (window_size, num_channels),
    where num_channels = 225 and the right hand occupies 63 channels.
    """
    both_hand_start = 33 * 3  # (33 pose + 21 left hand) * 3 coordinates
    both_hand_end = both_hand_start + 126  # 21 landmarks * 3 coordinates
    return sequence_data[:,:, both_hand_start:both_hand_end]
 
class SignumDataset(Dataset):
    def __init__(self, data_path, split):
        self.X = np.load(os.path.join(data_path, f'X_{split}.npy'))
        self.y = np.load(os.path.join(data_path, f'y_{split}.npy'))
        print(self.X.shape)
        self.X = extract_both_hand(self.X)
        print(self.X.shape)
   
    def __len__(self):
        return len(self.X)
   
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
 
def get_dataloader(data_path, split, batch_size):
    dataset = SignumDataset(data_path, split)
   
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
# Create train, validation, and test loaders
def get_all_dataloaders(data_path, batch_size):
    train_loader = get_dataloader(data_path, 'train', batch_size)
    val_loader = get_dataloader(data_path, 'val', batch_size)
    test_loader = get_dataloader(data_path, 'test', batch_size)
    return train_loader, val_loader, test_loader



class DistillationNPYDataset(Dataset):
    """Dataset that loads precomputed skeleton data from separate .npy files and includes DTW distances."""
    def __init__(self, folder: str, split: str, dtw_distances: np.ndarray):
        """
        Args:
            folder (str): Path to the folder containing .npy files.
            split (str): One of ['train', 'val', 'test'] to load the corresponding dataset.
            dtw_distances (np.ndarray): Precomputed DTW distance matrix.
        """
        self.X = np.load(os.path.join(folder, f'X_{split}.npy'))  # Load features
        self.y = np.load(os.path.join(folder, f'y_{split}.npy'))  # Load labels
        self.dtw_distances = dtw_distances

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int):
        raw_X = self.X[idx]  # Original shape: (num_frames, num_channels)
        both_hand_X = extract_both_hand(raw_X)  # Shape: (num_frames, 63)
        
        sample = {
            'X': torch.tensor(both_hand_X, dtype=torch.float32),  
            'y': torch.tensor(self.y[idx], dtype=torch.long),    
            'dtw_distances': torch.tensor(self.dtw_distances[idx], dtype=torch.float32)
        }
        return sample

def get_distillation_dataloader_from_folder(folder: str, dtw_distances: np.ndarray, 
                                            batch_size: int = 16, shuffle: bool = False, 
                                            num_workers: int = 4):
    """Creates dataloaders from a given folder with precomputed NPY data and DTW distances."""
    
    train_dataset = DistillationNPYDataset(folder, 'train', dtw_distances)
    val_dataset = DistillationNPYDataset(folder, 'val', dtw_distances)
    test_dataset = DistillationNPYDataset(folder, 'test', dtw_distances)
    
    def distillation_collate(batch):
        """Custom collate function for batches with DTW distances."""
        X = torch.stack([item['X'] for item in batch])
        y = torch.tensor([item['y'] for item in batch])
        dtw_distances = torch.tensor(np.stack([item['dtw_distances'] for item in batch]))
        return {'X': X, 'y': y, 'dtw_distances': dtw_distances}
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, collate_fn=distillation_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=distillation_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, collate_fn=distillation_collate)
    
    return train_loader, val_loader, test_loader

