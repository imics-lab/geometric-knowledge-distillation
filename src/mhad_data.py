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

class SkeletonDataset(Dataset):
    def __init__(self, data_path, split):
        self.X = np.load(os.path.join(data_path, f'X_{split}.npy'),  allow_pickle=True)
        self.y = np.load(os.path.join(data_path, f'y_{split}.npy'),  allow_pickle=True)
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[2], self.X.shape[1])
        self.y = np.where(self.y == 27, 0, self.y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def get_dataloader(data_path, split, batch_size):
    dataset = SkeletonDataset(data_path, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Create train, validation, and test loaders
def get_all_dataloaders(data_path, batch_size):
    train_loader = get_dataloader(data_path, 'train', batch_size)
    test_loader = get_dataloader(data_path, 'test', batch_size)
    return train_loader, test_loader

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
        print(self.X.shape)
        self.dtw_distances = dtw_distances
        self.X = self.X.reshape(self.X.shape[0], self.X.shape[2], self.X.shape[1])
        self.y = np.where(self.y == 27, 0, self.y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int):
               
        sample = {
            'X': torch.tensor(self.X[idx], dtype=torch.float32),  
            'y': torch.tensor(self.y[idx], dtype=torch.long),    
            'dtw_distances': torch.tensor(self.dtw_distances[idx], dtype=torch.float32)
        }
        return sample

def get_distillation_dataloader_from_folder(folder: str, dtw_distances: np.ndarray, 
                                            batch_size: int = 16, shuffle: bool = False, 
                                            num_workers: int = 4):
    """Creates dataloaders from a given folder with precomputed NPY data and DTW distances."""
    
    train_dataset = DistillationNPYDataset(folder, 'train', dtw_distances)
    test_dataset = DistillationNPYDataset(folder, 'test', dtw_distances)
    
    def distillation_collate(batch):
        """Custom collate function for batches with DTW distances."""
        X = torch.stack([item['X'] for item in batch])
        y = torch.tensor([item['y'] for item in batch])
        dtw_distances = torch.tensor(np.stack([item['dtw_distances'] for item in batch]))
        return {'X': X, 'y': y, 'dtw_distances': dtw_distances}
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, collate_fn=distillation_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, collate_fn=distillation_collate)
    
    return train_loader, test_loader